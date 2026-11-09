def test_package_importable_with_version() -> None:
    import agentfw

    assert agentfw.__version__ == "0.1.0"
