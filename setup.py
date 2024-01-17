from distutils.core import setup

setup(
  name = 'Simple_Genetic_Algorithm',         # How you named your package folder (MyLib)
  packages = ['Simple_Genetic_Algorithm'],   # Chose the same as "name"
  version = '0.1',      # Start with a small number and increase it with every change you make
  license='MPL-2.0',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Genetic Algorithm Framework',   # Give a short description about your library
  author = 'Tobia Ippolito',                   # Type in your name
  url = 'https://github.com/xXAI-botXx/Genetic-Algorithm',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/xXAI-botXx/Genetic-Algorithm/archive/v_01.tar.gz',    
  keywords = ['Optimization', 'Genetic-Algorithm', 'Hyperparameter-Tuning'],   # Keywords that define your package best
  install_requires=[            # used libraries
          'joblib'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MPL-2.0',   # Again, pick a license
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',      # Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12'
  ],
)


