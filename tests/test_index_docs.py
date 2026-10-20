import pytest

from src.index_docs import remove_stale_ids


class FakeCollection:
    def __init__(self, ids):
        self.ids = list(ids)
        self.deleted = []

    def get(self, include):
        assert include == []
        return {"ids": self.ids}

    def delete(self, ids):
        self.deleted.extend(ids)
        self.ids = [chunk_id for chunk_id in self.ids if chunk_id not in ids]


def test_remove_stale_ids_preserves_expected_records_and_batches_deletes():
    collection = FakeCollection(["keep-1", "old-1", "keep-2", "old-2", "old-3"])

    removed = remove_stale_ids(collection, {"keep-1", "keep-2"}, batch_size=2)

    assert removed == 3
    assert collection.deleted == ["old-1", "old-2", "old-3"]
    assert collection.ids == ["keep-1", "keep-2"]


def test_remove_stale_ids_rejects_invalid_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        remove_stale_ids(FakeCollection([]), set(), batch_size=0)
