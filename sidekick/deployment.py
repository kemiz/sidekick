from itertools import islice
from typing import Any, Dict, Generator, Iterable, List

import requests
from requests.adapters import HTTPAdapter

from .encode import DataItem, decode_feature, encode_feature

PredictData = Dict[str, List[Dict[str, Any]]]


def prediction_request(items: Iterable[DataItem],
                       dtypes: Dict[str, str]) -> PredictData:
    rows = list()
    for item in items:
        row = dict()
        for feature_name, dtype_str in dtypes.items():
            if feature_name not in item:
                raise ValueError('Item is missing feature: %s' % feature_name)
            feature = item[feature_name]
            row[feature_name] = encode_feature(feature, dtype_str)
        rows.append(row)
    return {'rows': rows}


def parse_prediction(
        data: PredictData,
        dtypes: Dict[str, str]) -> Generator[DataItem, None, None]:
    if 'errorCode' in data:
        raise IOError(
            '%s: %s' % (data['errorCode'], data.get('errorMessage', '')))
    if 'rows' not in data:
        raise ValueError('Return data does not contain rows')

    for row in data['rows']:
        item = dict()
        for feature_name, dtype_str in dtypes.items():
            if feature_name not in row:
                raise ValueError('Item is missing feature: %s' % feature_name)
            item[feature_name] = decode_feature(row[feature_name], dtype_str)
        yield item


class Deployment:
    """Sidekick for Peltarion platform deployments
    """
    BATCH_SIZE = 128
    MAX_RETRIES = 3

    def __init__(self,
                 url: str,
                 token: str,
                 dtypes_in: Dict[str, str],
                 dtypes_out: Dict[str, str]):
        self._dtypes_in = dtypes_in
        self._dtypes_out = dtypes_out
        self._headers = {'Authorization': 'Bearer ' + token}
        self._url = url
        self._session = requests.Session()
        self._session.mount('', HTTPAdapter(max_retries=self.MAX_RETRIES))
        self._session.headers.update({'User-Agent': 'sidekick'})

    def predict_lazy(self, items: Iterable[DataItem]) -> \
            Generator[DataItem, None, None]:
        iterator = iter(items)
        while True:
            batch = list(islice(iterator, self.BATCH_SIZE))
            if not batch:
                break

            encoded = prediction_request(batch, self._dtypes_in)
            response = self._session.post(
                url=self._url,
                headers=self._headers,
                json=encoded
            )
            response.raise_for_status()  # Raise exceptions
            yield from parse_prediction(response.json(), self._dtypes_out)

    def predict_many(self, items: Iterable[DataItem]) -> List[DataItem]:
        return list(self.predict_lazy(items))

    def predict(self, **item: DataItem) -> DataItem:
        return self.predict_many([item])[0]
