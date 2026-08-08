"""Minimal CLI tests - just enough to catch wiring errors."""

from typer.testing import CliRunner

from substratum.cli.bass import app

runner = CliRunner()


def test_render_writes_wav(tmp_path):
    out = tmp_path / "out.wav"
    result = runner.invoke(app, ["--freq", "38", "--output", str(out)])
    assert result.exit_code == 0
    assert out.exists()


def test_presets_command_lists_all(tmp_path):
    result = runner.invoke(app, ["presets"])
    assert result.exit_code == 0
    assert "velvet" in result.stdout
    assert "earthquake" in result.stdout


def test_unknown_subcommand_fails():
    result = runner.invoke(app, ["frobnicate"])
    assert result.exit_code != 0


def test_gui_command_has_help():
    result = runner.invoke(app, ["gui", "--help"])
    assert result.exit_code == 0
    assert "Textual GUI" in result.stdout
