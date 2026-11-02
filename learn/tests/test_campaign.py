import importlib.util
import os
import subprocess

import pytest

SPEC = importlib.util.spec_from_file_location('campaign', os.path.join(os.path.dirname(__file__), '..', 'evolve', 'campaign.py'))


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    """The sync module pointed at a local bare repo (standing in for GitHub) and a scratch evolve/ folder."""
    remote = tmp_path / 'remote.git'
    subprocess.run(['git', 'init', '-q', '--bare', '-b', 'main', str(remote)], check=True)
    def at(name):
        module = importlib.util.module_from_spec(SPEC)
        SPEC.loader.exec_module(module)
        here = tmp_path / name
        (here / 'chunks' / 'optim').mkdir(parents=True)
        (here / 'chunks' / 'optim' / 'baseline_adamw.py').write_text('baseline\n')
        monkeypatch.setattr(module, 'HERE', str(here))
        monkeypatch.setattr(module, 'CLONE', str(here / '.campaign'))
        monkeypatch.setattr(module, 'REMOTE', str(remote))
        return module, here
    for var in ('GIT_AUTHOR_NAME', 'GIT_COMMITTER_NAME'):
        monkeypatch.setenv(var, 'test')
    for var in ('GIT_AUTHOR_EMAIL', 'GIT_COMMITTER_EMAIL'):
        monkeypatch.setenv(var, 'test@example.invalid')
    return at


def test_a_campaign_pushed_from_one_place_restores_in_another(campaign):
    a, here_a = campaign('a')
    (here_a / 'chunks' / 'optim' / 'cand1.py').write_text('candidate one\n')
    (here_a / 'archive').mkdir()
    (here_a / 'archive' / 'genomes.jsonl').write_text('{"id": "gen0"}\n')
    (here_a / 'archive' / 'private').mkdir()
    (here_a / 'archive' / 'private' / 'secret.jsonl').write_text('holdout\n')
    a.push()
    b, here_b = campaign('b')
    b.pull()
    assert (here_b / 'chunks' / 'optim' / 'cand1.py').read_text() == 'candidate one\n'
    assert (here_b / 'archive' / 'genomes.jsonl').read_text() == '{"id": "gen0"}\n'
    assert not (here_b / 'archive' / 'private').exists()  # the engine's private store stays local
    assert 'baseline_adamw.py' not in os.listdir(here_b / '.campaign' / 'chunks' / 'optim')  # baselines live in the main repo


def test_impls_are_append_only_and_the_archive_only_grows(campaign):
    a, here_a = campaign('a')
    (here_a / 'chunks' / 'optim' / 'cand1.py').write_text('v1\n')
    (here_a / 'archive').mkdir()
    (here_a / 'archive' / 'genomes.jsonl').write_text('r1\n')
    a.push()
    b, here_b = campaign('b')
    b.pull()
    (here_b / 'archive' / 'genomes.jsonl').write_text('r1\nr2\n')  # b scored another candidate
    b.push()
    (here_a / 'chunks' / 'optim' / 'cand1.py').write_text('v2\n')  # a edited an archived impl
    with pytest.raises(SystemExit):
        a.push()
    (here_a / 'chunks' / 'optim' / 'cand1.py').write_text('v1\n')
    a.pull()
    assert (here_a / 'archive' / 'genomes.jsonl').read_text() == 'r1\nr2\n'
    (here_a / 'archive' / 'genomes.jsonl').write_text('r1\nr3\n')  # diverged from the repo's r1 r2
    with pytest.raises(SystemExit):
        a.push()
