"""
train.py's `_git_sha`/`_short_sha` -- reproducibility depends on the logged SHA
actually matching what ran. A dirty working tree must be visible, not silently
tagged with the previous commit's SHA as if it were reproducible from it.
"""

from __future__ import annotations

import subprocess

from lsm.train import _git_sha, _short_sha


def _run(*args, cwd):
    subprocess.run(args, cwd=cwd, check=True, capture_output=True)


def _init_repo(path):
    _run("git", "init", cwd=path)
    _run("git", "config", "user.email", "test@example.com", cwd=path)
    _run("git", "config", "user.name", "Test", cwd=path)


def test_git_sha_on_a_clean_committed_repo_is_the_bare_sha(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "a.txt").write_text("hello")
    _run("git", "add", "a.txt", cwd=tmp_path)
    _run("git", "commit", "-m", "initial", cwd=tmp_path)

    sha = _git_sha(cwd=tmp_path)
    assert not sha.endswith("-dirty")
    assert len(sha) == 40  # a real SHA, not "uncommitted"


def test_git_sha_on_a_dirty_tree_is_suffixed(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "a.txt").write_text("hello")
    _run("git", "add", "a.txt", cwd=tmp_path)
    _run("git", "commit", "-m", "initial", cwd=tmp_path)

    (tmp_path / "a.txt").write_text("modified, not committed")
    sha = _git_sha(cwd=tmp_path)
    assert sha.endswith("-dirty")


def test_git_sha_dirty_flag_clears_once_committed(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "a.txt").write_text("hello")
    _run("git", "add", "a.txt", cwd=tmp_path)
    _run("git", "commit", "-m", "initial", cwd=tmp_path)
    (tmp_path / "a.txt").write_text("modified")
    assert _git_sha(cwd=tmp_path).endswith("-dirty")

    _run("git", "add", "a.txt", cwd=tmp_path)
    _run("git", "commit", "-m", "second", cwd=tmp_path)
    assert not _git_sha(cwd=tmp_path).endswith("-dirty")


def test_git_sha_with_no_commits_yet_is_uncommitted(tmp_path):
    _init_repo(tmp_path)
    assert _git_sha(cwd=tmp_path) == "uncommitted"


def test_git_sha_outside_any_repo_is_uncommitted(tmp_path):
    assert _git_sha(cwd=tmp_path) == "uncommitted"


def test_short_sha_truncates_a_clean_sha_to_eight_chars():
    assert _short_sha("abcdef0123456789") == "abcdef01"


def test_short_sha_preserves_dirty_suffix_after_truncating():
    assert _short_sha("abcdef0123456789-dirty") == "abcdef01-dirty"


def test_short_sha_passes_through_uncommitted_unchanged():
    # "uncommitted" is shorter than 8 chars and has no -dirty suffix -- just
    # itself, truncation is a no-op here.
    assert _short_sha("uncommitted") == "uncommit"
