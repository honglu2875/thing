import subprocess


def test_exits():
    process = subprocess.Popen(
        'python -c "import thing;thing.serve();exit()"',
        shell=True,
        stdout=subprocess.PIPE,
    )
    process.wait(timeout=2.0)
    assert process.returncode == 0
