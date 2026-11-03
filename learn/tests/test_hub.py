import pytest

from pet_learn import hub


def test_the_token_comes_from_the_environment_or_the_runpod_env_file(tmp_path, monkeypatch):
    monkeypatch.setenv('HF_TOKEN', 'from-env')
    assert hub.token() == 'from-env'
    monkeypatch.delenv('HF_TOKEN')
    env_file = tmp_path / 'runpod.env'
    env_file.write_text('RUNPOD_API_KEY=x\nexport HF_TOKEN="from-file"\n')
    monkeypatch.setenv('RUNPOD_ENV_FILE', str(env_file))
    assert hub.token() == 'from-file'
    env_file.write_text('RUNPOD_API_KEY=x\n')
    with pytest.raises(SystemExit):
        hub.token()


def test_the_dataset_card_header_is_valid_yaml():
    import yaml
    header = hub.CARD.split('---')[1]
    assert yaml.safe_load(header)['pretty_name'].startswith('TurtleBot 4')


def test_a_pod_fetches_only_the_searched_sessions_however_many_are_pinned(tmp_path):
    import json
    source = {'files': ['a.npz', 'a.maps.npz', 'b.npz', 'b.maps.npz', 'new.npz', 'new.maps.npz']}
    manifest = tmp_path / 'm.json'
    manifest.write_text(json.dumps({'sessions': {'a.npz': 'x', 'b.npz': 'y'}}))
    assert hub.wanted(source, str(manifest)) == ['a.maps.npz', 'a.npz', 'b.maps.npz', 'b.npz']
    assert hub.wanted(source) == source['files']
    manifest.write_text(json.dumps({'sessions': {'gone.npz': 'z'}}))
    with pytest.raises(SystemExit):
        hub.wanted(source, str(manifest))


def test_every_folder_with_a_bag_is_backed_up_even_one_not_stopped_cleanly(tmp_path):
    for name, files in {'a': ['a_0.mcap', 'metadata.yaml'], 'b': ['b_0.mcap'], 'c': ['metadata.yaml'],
                        'd': ['d_0.mcap', 'd_1.mcap', 'metadata.yaml', 'notes.txt']}.items():
        (tmp_path / name).mkdir()
        for f in files:
            (tmp_path / name / f).write_text('x')
    (tmp_path / 'stray.mcap').write_text('x')
    assert hub.bag_sessions(str(tmp_path)) == ['a', 'b', 'd']


def test_a_restore_fetches_exactly_the_bags_asked_for():
    repo = ['README.md', '.gitattributes', 'a/a_0.mcap', 'a/metadata.yaml', 'b/b_0.mcap', 'b/b_1.mcap', 'b/metadata.yaml']
    assert hub.bag_files(repo) == sorted(repo[2:])
    assert hub.bag_files(repo, ['b']) == ['b/b_0.mcap', 'b/b_1.mcap', 'b/metadata.yaml']
    assert hub.bag_files(repo, ['zz']) == []


def test_the_drives_card_header_is_valid_yaml():
    import yaml
    assert yaml.safe_load(hub.DRIVES_CARD.split('---')[1])['pretty_name'].startswith('TurtleBot 4')
