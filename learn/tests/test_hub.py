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
