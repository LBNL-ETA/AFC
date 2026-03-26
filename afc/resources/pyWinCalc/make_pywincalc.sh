mkdir compile_pywincalc
cd compile_pywincalc
git clone https://github.com/LBNL-ETA/pyWinCalc
cd pyWinCalc
# git checkout EffectiveThermalMultipliers
# USE ID INSTEAD
git checkout 7514f9e
sed -i 's/EffectiveThermalValues/v2.6.2/g' CMakeLists-WinCalc.txt.in
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install build
CXXFLAGS="-Wno-error=missing-field-initializers" python3 -m build