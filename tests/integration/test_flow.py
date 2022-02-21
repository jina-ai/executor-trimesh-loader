from pathlib import Path

import pytest
from jina import Document, DocumentArray, Flow

from executor import TrimeshLoader

data_dir = Path(__file__).parent.parent


@pytest.fixture
def docs():
    docs = DocumentArray()
    doc = Document(
        uri='https://storage.googleapis.com/showcase-3d-models/ShapeNetV2/airplane_aeroplane_plane_0.glb'
    )
    docs.append(doc)
    docs.append(Document(uri=open(data_dir / 'test.base64').read()))

    return docs


def test_use_in_flow(docs):
    with Flow().add(uses=TrimeshLoader, name='trimesh_loader') as flow:
        data = flow.post(on='/index', inputs=docs, return_results=True)
        for doc in data:
            assert doc.tensor is not None
            assert doc.tensor.shape == (1024, 3)
