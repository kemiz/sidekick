import random

import numpy as np
import pytest
import responses
from PIL import Image

import sidekick
from sidekick import Deployment


@responses.activate
def test_deployment_scalars():
    responses.add(
        responses.POST,
        'http://deployment_url',
        json={'rows': [{'out': 1.0}]}
    )
    deployment = Deployment(
        url='http://deployment_url',
        token='deployment_token',
        dtypes_in={'feature_1': 'Float (1)', 'feature_2': 'Int (1)'},
        dtypes_out={'out': 'Float (1)'}
    )

    # Single scalar value prediction
    prediction = deployment.predict(feature_1=1.0, feature_2=1)
    assert prediction == {'out': 1.0}

    # List of scalar value prediction
    prediction = deployment.predict(feature_1=1.0, feature_2=1)
    assert prediction == {'out': 1.0}

    # Generator of scalar value predictions
    prediction = deployment.predict(feature_1=1.0, feature_2=1)
    assert prediction == {'out': 1.0}

    # Send bad type
    with pytest.raises(TypeError):
        deployment.predict(feature_1=1.0, feature_2='foo')


@responses.activate
def test_deployment_numpy_autoencoder():
    arr = np.random.rand(100, 10, 3).astype(np.float32)
    encoded = sidekick.encode.NumpyEncoder().encode_json(arr)
    responses.add(
        responses.POST,
        'http://deployment_url',
        json={'rows': [{'numpy_out': encoded}]}
    )
    deployment = Deployment(
        url='http://deployment_url',
        token='deployment_token',
        dtypes_in={'numpy_in': 'Numpy (100x10x3)'},
        dtypes_out={'numpy_out': 'Numpy (100x10x3)'}
    )

    # Single numpy prediction
    prediction = deployment.predict(numpy_in=arr)
    np.testing.assert_array_equal(prediction['numpy_out'], arr)

    # List of numpy predictions
    predictions = deployment.predict_many({'numpy_in': arr} for _ in range(10))
    for prediction in predictions:
        np.testing.assert_array_equal(prediction['numpy_out'], arr)

    # Generator of numpy predictions
    predictions = deployment.predict_lazy({'numpy_in': arr} for _ in range(10))
    for prediction in predictions:
        np.testing.assert_array_equal(prediction['numpy_out'], arr)

    # Send bad shape
    with pytest.raises(ValueError):
        deployment.predict(numpy_in=np.random.rand(100, 1, 1))


@responses.activate
def test_deployment_image_autoencoder():
    arr = np.uint8(np.random.rand(100, 10, 3) * 255)
    image = Image.fromarray(arr)
    image.format = 'png'

    encoded = sidekick.encode.ImageEncoder().encode_json(image)
    responses.add(
        responses.POST,
        'http://deployment_url',
        json={'rows': [{'image_out': encoded}]}
    )
    deployment = Deployment(
        url='http://deployment_url',
        token='deployment_token',
        dtypes_in={'image_in': 'Image (100x10x3)'},
        dtypes_out={'image_out': 'Image (100x10x3)'}
    )

    # Single image prediction
    prediction = deployment.predict(image_in=image)
    np.testing.assert_array_equal(np.array(prediction['image_out']), arr)

    # Send bad type
    with pytest.raises(TypeError):
        deployment.predict(image_in=arr)


@responses.activate
def test_deployment_categorical():
    predictions = {str(i): random.random() for i in range(10)}
    responses.add(
        responses.POST,
        'http://deployment_url',
        json={'rows': [{'label': predictions}]}
    )
    deployment = Deployment(
        url='http://deployment_url',
        token='deployment_token',
        dtypes_in={'feature_1': 'Float (1)', 'feature_2': 'Numpy (30)'},
        dtypes_out={'label': 'Categorical (10)'}
    )

    # Single scalar value prediction
    prediction = deployment.predict(
        feature_1=1.0, feature_2=np.random.rand(30))
    assert prediction == {'label': predictions}

    # Send bad type
    with pytest.raises(TypeError):
        deployment.predict(feature_1=1.0, feature_2='foo')

    # Return bad shape
    deployment = Deployment(
        url='http://deployment_url',
        token='deployment_token',
        dtypes_in={'feature_1': 'Float (1)', 'feature_2': 'Numpy (30)'},
        dtypes_out={'label': 'Categorical (5)'}
    )
    with pytest.raises(ValueError):
        deployment.predict(feature_1=0.0, feature_2=np.random.rand(30))
