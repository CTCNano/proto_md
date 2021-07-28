"""
Basic sanity test for the proto_md package.
"""
import pytest
import proto_md


def test_proto_md_imported():
    """Sample test, will always pass so long as import statement worked"""
    import sys

    assert "proto_md" in sys.modules
