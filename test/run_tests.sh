pip3 -qq install pylint flake8 pytest

echo "Running pylint"
#pylint $(find .. -name "*.py") --disable=C,R
pylint --fail-under=8 $(find .. -name "*.py" -not -path "../build/*") --disable=C,R

echo "Running flake8"
# stop the build if there are Python syntax errors or undefined names
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
# exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
#flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

echo "Running pytest"
pytest