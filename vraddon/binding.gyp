{
  'targets': [
    {
      'target_name': 'vrviewer',
      'sources': [ 'binding.cpp' ],
      "libraries": ["../3dconnexion/siappMT.lib", "Comctl32.lib"],
	  "include_dirs" : [
		"<!(node -e \"require('nan')\")"
		]
    }
  ]
}