import abc
import base64
import io
import itertools
from numbers import Real
from typing import Any, Mapping, Set, Tuple

import numpy as np
from PIL import Image

DataItem = Mapping[str, Any]


class Encoder(abc.ABC):

    def file_extension(self, value):
        _ = value
        return ''

    @abc.abstractmethod
    def expects(self):
        pass

    def encode_json(self, value):
        return self.encode(value)

    def decode_json(self, encoded):
        return self.decode(encoded)

    @abc.abstractmethod
    def encode(self, value):
        pass

    @abc.abstractmethod
    def decode(self, encoded):
        pass

    @abc.abstractmethod
    def verify(self, value, shape: Tuple[int, ...]):
        pass


class BinaryEncoder(Encoder):

    @abc.abstractmethod
    def media_type(self, value) -> str:
        pass

    @abc.abstractmethod
    def encode(self, value) -> bytes:
        pass

    @abc.abstractmethod
    def decode(self, encoded: bytes):
        pass

    def encode_json(self, value) -> str:
        return 'data:%s;base64,%s' % (
            self.media_type(value),
            base64.b64encode(self.encode(value)).decode()
        )

    def decode_json(self, encoded: str) -> Any:
        try:
            data_type, b64_data = encoded.split(',', 1)
            _, media_description = data_type.split(':', 1)
            media_type, _ = media_description.split(';', 1)
        except ValueError:
            raise ValueError('Not a valid Data URL')
        data = base64.b64decode(b64_data)
        value = self.decode(data)
        expected_media_type = self.media_type(value)
        if media_type != expected_media_type:
            raise ValueError('Not a valid media type, expected %s but got %s' %
                             (expected_media_type, media_type))
        return value


class CategoricalEncoder(Encoder):

    def expects(self) -> Set:
        return {dict}

    def verify(self, value: dict, shape: Tuple[int, ...]):
        shape = tuple(shape)
        if len(shape) != 1:
            raise ValueError(
                'Required shape: %s but Categorical must be 1-D' % shape)
        if len(value) != shape[0]:
            raise ValueError('Categorical expected %i values, got: %i'
                             % (shape[0], len(value)))

    @staticmethod
    def check_type(value):
        if not isinstance(value, dict):
            raise TypeError('Expected %s but received %s' %
                            (dict, type(value)))

    @classmethod
    def encode(cls, value):
        cls.check_type(value)
        return value

    @classmethod
    def decode(cls, encoded):
        cls.check_type(encoded)
        return encoded


class ScalarEncoder(Encoder):

    def __init__(self, scalar_type):
        self._scalar_type = scalar_type

    def expects(self) -> Set:
        return {self._scalar_type}

    def verify(self, scalar: Real, shape: Tuple[int, ...]):
        shape = tuple(shape)
        if shape != (1, ):
            raise ValueError('Required shape: %s but Float must be scalar'
                             % shape)

    def check_type(self, value):
        if not isinstance(value, self._scalar_type):
            raise TypeError('Expected %s but received %s' %
                            (self._scalar_type, type(value)))

    def encode(self, value):
        self.check_type(value)
        return value

    def decode(self, encoded):
        self.check_type(encoded)
        return encoded


class NumpyEncoder(BinaryEncoder):

    @classmethod
    def file_extension(cls, value):
        return 'npy'

    def media_type(self, value):
        return 'application/x.peltarion.npy'

    def expects(self) -> Set:
        return {np.ndarray}

    @classmethod
    def verify(cls, value: np.ndarray, shape: Tuple[int, ...]):
        if not isinstance(value, np.ndarray):
            raise TypeError('Expected numpy array but received %s'
                            % type(value))
        shape = tuple(shape)
        if value.shape != shape:
            raise ValueError('Expected shape: %s, numpy array has shape: %s'
                             % (shape, value.shape))

    @classmethod
    def encode(cls, value: np.ndarray) -> bytes:
        value = value.astype(np.float32)
        with io.BytesIO() as buffer:
            np.save(buffer, value)
            return buffer.getvalue()

    @classmethod
    def decode(cls, encoded: bytes) -> np.ndarray:
        with io.BytesIO(encoded) as buffer:
            array = np.load(buffer)
        return array


class ImageEncoder(BinaryEncoder):

    def file_extension(self, value):
        if value.format is None:
            raise ValueError('No format set on image, please specify '
                             '(see the documentation for details)')
        return value.format.lower()

    def media_type(self, value):
        if value.format is None:
            raise ValueError('No format set on image, please specify manually')
        return 'image/' + value.format.lower()

    def expects(self) -> Set:
        return {Image.Image}

    @classmethod
    def verify(cls, image: Image, shape: Tuple[int, ...]):
        if not isinstance(image, Image.Image):  # This really is the type...
            raise TypeError('Expected an Image but received %s' % type(image))

        shape = tuple(shape)
        channels = image.getbands()
        spatial_shape = tuple(reversed(image.size))
        feature_shape = (*spatial_shape, len(channels))
        if feature_shape != shape:
            raise ValueError('Expected shape: %s, but Image has shape: %s'
                             % (shape, feature_shape))

    @classmethod
    def encode(cls, value: Image) -> bytes:
        original_format = value.format
        # We do not support 4-channel PNGs or alpha in general
        if value.mode == 'RGBA':
            value = value.convert('RGB')
        elif value.mode == 'LA':
            value = value.convert('L')
        with io.BytesIO() as image_bytes:
            value.save(image_bytes, format=original_format)
            return image_bytes.getvalue()

    @classmethod
    def decode(cls, encoded: bytes):
        with io.BytesIO(encoded) as buffer:
            image = Image.open(buffer)
            image.load()
        return image


DTYPE_ENCODERS = {
    'Float': ScalarEncoder(float),
    'Int': ScalarEncoder(int),
    'Categorical': CategoricalEncoder(),
    'Numpy': NumpyEncoder(),
    'Image': ImageEncoder()
}


DTYPE_COMPATIBILITY = {
    dtype_name: encoder.expects()
    for dtype_name, encoder in DTYPE_ENCODERS.items()
}


ENCODER_COMPATIBILITY = dict(itertools.chain.from_iterable(
    ((compatible_type, encoder) for compatible_type in encoder.expects())
    for encoder in DTYPE_ENCODERS.values()
))


FILE_EXTENSION_ENCODERS = {
    'npy': DTYPE_ENCODERS['Numpy'],
    'png': DTYPE_ENCODERS['Image'],
    'jpg': DTYPE_ENCODERS['Image'],
    'jpeg': DTYPE_ENCODERS['Image']
}


def parse_dtype(dtype_str: str) -> Tuple[str, Tuple[int, ...]]:
    dtype_name, dtype_shape_str = dtype_str.strip().split(' ', 1)
    shape = tuple(int(dim) for dim in dtype_shape_str.strip('()').split('x'))
    if not shape or not all(dim > 0 for dim in shape):
        raise ValueError('Bad shape information provided. Please verify the '
                         'full type string including shape was used')
    if dtype_name not in DTYPE_ENCODERS:
        raise TypeError('This type is currently not supported: %s'
                        % dtype_name)
    return dtype_name, shape


def encode_feature(feature, dtype_str: str) -> Any:
    dtype_name, shape = parse_dtype(dtype_str)
    encoder = DTYPE_ENCODERS[dtype_name]
    encoder.verify(feature, shape)
    return encoder.encode_json(feature)


def decode_feature(feature, dtype_str: str) -> Any:
    dtype_name, shape = parse_dtype(dtype_str)
    encoder = DTYPE_ENCODERS[dtype_name]
    decoded = encoder.decode_json(feature)
    encoder.verify(decoded, shape)
    return decoded
