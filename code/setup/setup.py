import os

def notes_home_dir():
    noteshomedir = 'C:\\Users\\h2_sf\\Documents\\Magic Briefcase\\a304s18\\'
    if not os.path.exists(noteshomedir):
        raise Exception('something is very wrong: %s does not exist'%noteshomedir)
    return noteshomedir

def code_home_dir():
    codehomedir = os.path.join(notes_home_dir(), 'code/')
    return codehomedir

def data_home_dir():
    datahomedir = os.path.join(notes_home_dir(), 'data/')
    if not os.path.exists(datahomedir):
        os.makedirs(datahomedir)
    return datahomedir

def image_home_dir():
    imghomedir = os.path.join(notes_home_dir(), 'img/')
    if not os.path.exists(imghomedir):
        os.makedirs(imghomedir)
    return imghomedir
    
def sdss_filter_dir():
    filterdir = os.path.join(data_home_dir(), 'sdss_filters/')
    if not os.path.exists(filterdir):
        os.makedirs(filterdir)
    return filterdir
    
if __name__ == '__main__':
    sdssfilterdir = sdss_filter_dir()
    datahomedir = data_home_dir()
    imghomedir = image_home_dir()

    import py_compile
    py_compile.compile(os.path.join(code_home_dir(),'setup/setup.py'))
