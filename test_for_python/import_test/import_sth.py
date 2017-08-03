import module_to_import
dd=module_to_import.cc
print dd
print 'hhh'

reload(module_to_import)
print dd
dd=module_to_import.cc
print dd
