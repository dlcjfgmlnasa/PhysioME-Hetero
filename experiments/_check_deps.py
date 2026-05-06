"""Check which dependencies our code actually needs and report install status."""
import importlib

REQUIRED = [
    # Core
    ('numpy', None),
    ('scipy', None),
    ('torch', None),
    ('torchvision', None),
    # Models / training
    ('timm', None),
    ('einops', None),
    # SSL / metrics
    ('sklearn', 'scikit-learn'),
    ('mne', None),
    ('tensorboard', None),
    # Data parsers
    ('vitaldb', None),
    ('wfdb', None),
    # Misc
    ('yaml', 'PyYAML'),
    ('tqdm', None),
    ('matplotlib', None),
]

print('module                 status         (pip name)')
print('-' * 60)
missing = []
for mod, pip_name in REQUIRED:
    pip_str = pip_name or mod
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, '__version__', '?')
        print(f'  {mod:<22} OK   ({ver})')
    except ImportError:
        print(f'  {mod:<22} MISSING')
        missing.append(pip_str)

if missing:
    print()
    print('To install missing:')
    print('  pip install ' + ' '.join(missing))
else:
    print()
    print('All required packages installed.')
