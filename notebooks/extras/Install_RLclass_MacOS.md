# Install RL class MacOS
## 1. Visualization
```
conda install graphviz python-graphviz
```

## 2. Install required librairies
You can install required packages `cmake` and `zlib` with `brew`:
```
brew install cmake zlib
```

__WARNING__: `brew` auto-update all packages when doing an install ! 
Use cmd below to not auto-update:
```
HOMEBREW_NO_AUTO_UPDATE=1 brew install cmake zlib
```
### BigSur 
It seems that on BigSur you need to force `pyglet==1.5.11`. See [pyglet](#pyglet) below. Just run:
```
pip install pyglet==1.5.11
```

## 3. Install `gym` and his environments with pip
```
pip install gym gym[atari] gym[classic_control] gym[box2d] gym[algorithms]
```

NOTE: If you use __zsh__ you need to add quotes arround `gym[env]` like below
```
pip install gym 'gym[atari]' 'gym[classic_control]' 'gym[box2d]' 'gym[algorithms]'
```

# Known Issues
### pyglet
You can resolve a problem with `pyglet` by forcing the install of  version `1.5.11`, run cmd below to do it:
```
pip install pyglet==1.5.11
```
See [here](https://github.com/openai/gym/issues/2101#issuecomment-730513761) for more information.

### gym env windows not closing
The `.close()` method is not working for some people when called from a code cell of a jupyter notebook but work in a python script `.py

# Sources
- Medium article: [Install OpenAI Gym with Atari on macOs](https://medium.com/@lyu.xueguang/install-openai-gym-with-atari-on-macos-cd35d09194ee)
- Issue on gym repo: [Gym on Mac OS X Big Sur #2101](https://github.com/openai/gym/issues/2101)