project('scikits-odes-daepack',
  ['c', 'fortran'],
  version : '0.1',
  meson_version: '>= 1.1.0',
  default_options : [
    'warning_level=1',
    'buildtype=release'
  ])

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

incdir_numpy = run_command(py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

incdir_f2py = run_command(py,
    ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true
).stdout().strip()

fortranobject_c = incdir_f2py / 'fortranobject.c'

py_pkg_name = 'scikits_odes_daepack'
py_pkg_dir = 'src' / py_pkg_name
daepack_dir = py_pkg_dir / 'daepack'

inc_np_all = include_directories(incdir_numpy, incdir_f2py)

ddaspk_source = custom_target('ddaspkmodule.c',
  input : [py_pkg_dir / 'ddaspk.pyf'],
  output : ['ddaspkmodule.c', 'ddaspk-f2pywrappers.f'],
  command : [py, '-m', 'numpy.f2py', '@INPUT@'],
)

lsodi_source = custom_target('lsodimodule.c',
  input : [py_pkg_dir / 'lsodi.pyf'],
  output : ['lsodimodule.c', 'lsodi-f2pywrappers.f'],
  command : [py, '-m', 'numpy.f2py', '@INPUT@'],
)

daepack_dep = declare_dependency(
  sources : [
    daepack_dir / 'ainvg.f',
    daepack_dir / 'cfode.f',
    daepack_dir / 'daux.f',
    daepack_dir / 'ddaspk.f',
    daepack_dir / 'dlinpk.f',
    daepack_dir / 'ewset.f',
    daepack_dir / 'intdy.f',
    daepack_dir / 'lsodi.f',
    daepack_dir / 'prepji.f',
    daepack_dir / 'solsy.f',
    daepack_dir / 'stodi.f',
    daepack_dir / 'vnorm.f',
    daepack_dir / 'xerrwv.f',
  ])

py.extension_module('ddaspk',
  [
    ddaspk_source,
    fortranobject_c,
  ],
  include_directories: [
    inc_np_all,
  ],
  dependencies : [
    py_dep,
    daepack_dep,
  ],
  install : true,
  subdir : py_pkg_name,
)

py.extension_module('lsodi',
  [
    lsodi_source,
    fortranobject_c,
  ],
  include_directories: [
    inc_np_all,
  ],
  dependencies : [
    py_dep,
    daepack_dep,
  ],
  install : true,
  subdir : py_pkg_name,
)

py.install_sources(
  [
    py_pkg_dir / '__init__.py',
    py_pkg_dir / 'ddaspkint.py',
    py_pkg_dir / 'lsodiint.py',
  ],
  subdir : py_pkg_name,
)
