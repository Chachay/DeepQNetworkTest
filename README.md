# Deep Q Network Test
Somewhat clone version of Deep Q Learning Reinforcement Learning demo in ConvNetJS project

<img src="https://raw.githubusercontent.com/Chachay/DeepQNetworkTest/master/demo.png">

## Requirement
- Python 2.7
- Chainer (1.15.0)
- numpy
- wxPython3.0-win32-py27

## Note
pyplot in wxPython3.0 contains a ciritical error. c.f.[Crash on pyplot demo \- Google Groups](https://groups.google.com/forum/#!msg/wxpython-users/VGBZ2Uiv864/BlWfCz_Q_mAJ)
You have to overwrite [wxPython/plot\.py](https://github.com/wxWidgets/wxPython/blob/master/wx/lib/plot.py)

## Usage
```
python DQN001.py
```

## References
- [Deep Q Learning Reinforcement Learning demo](http://cs.stanford.edu/people/karpathy/convnetjs/demo/rldemo.html)
- [倒立振子で学ぶ DQN (Deep Q Network)](http://qiita.com/ashitani/items/bb393e24c20e83e54577)
