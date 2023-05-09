import hashlib
import os
import sys
import tarfile
import zipfile
import requests
import os.path as osp

from urllib.request import urlopen
from urllib.parse import urlparse


class Downloader:
    MB = 1024*1024
    BUFSIZE = 10*MB

    def __init__(self, **kwargs):
        self._name = kwargs.pop('name')
        self._url = kwargs.pop('url', None)
        self._filename = kwargs.pop('filename')
        self._sha = kwargs.pop('sha', None)
        self._saveTo = kwargs.pop('saveTo', './data')
        self._extractTo = kwargs.pop('extractTo', './data')

    def __str__(self):
        return 'Downloader for <{}>'.format(self._name)

    def printRequest(self, r):
        def getMB(r):
            d = dict(r.info())
            for c in ['content-length', 'Content-Length']:
                if c in d:
                    return int(d[c]) / self.MB
            return '<unknown>'
        print('    {} {} [{} Mb]'.format(r.getcode(), r.msg, getMB(r)))

    def verifyHash(self):
        if not self._sha:
            return False
        sha = hashlib.sha1()
        try:
            with open(osp.join(self._saveTo, self._filename), 'rb') as f:
                while True:
                    buf = f.read(self.BUFSIZE)
                    if not buf:
                        break
                    sha.update(buf)
            if self._sha != sha.hexdigest():
                print('    actual {}'.format(sha.hexdigest()))
                print('    expect {}'.format(self._sha))
            return self._sha == sha.hexdigest()
        except Exception as e:
            print('    catch {}'.format(e))

    def get(self):
        print('  {}: {}'.format(self._name, self._filename))
        if self.verifyHash():
            print('    hash match - skipping download')
        else:
            basedir = os.path.dirname(self._saveTo)
            if basedir and not os.path.exists(basedir):
                print('    creating directory: ' + basedir)
                os.makedirs(basedir, exist_ok=True)

            print('    hash check failed - downloading')
            if 'drive.google.com' in self._url:
                urlquery = urlparse(self._url).query.split('&')
                for q in urlquery:
                    if 'id=' in q:
                        gid = q[3:]
                sz = GDrive(gid)(osp.join(self._saveTo, self._filename))
                print('    size = %.2f Mb' % (sz / (1024.0 * 1024)))
            else:
                print('    get {}'.format(self._url))
                self.download()

            # Verify hash after download
            print('    done')
            print('    file {}'.format(self._filename))
            if self.verifyHash():
                print('    hash match - extracting')
            else:
                print('    hash check failed - exiting')

        # Extract
        if '.zip' in self._filename:
            print('    extracting - ', end='')
            self.extract()
            print('done')

        return True

    def download(self):
        try:
            r = urlopen(self._url, timeout=60)
            self.printRequest(r)
            self.save(r)
        except Exception as e:
            print('  catch {}'.format(e))

    def extract(self):
        fileLocation = os.path.join(self._saveTo, self._filename)
        try:
            if self._filename.endswith('.zip'):
                with zipfile.ZipFile(fileLocation) as f:
                    for member in f.namelist():
                        path = osp.join(self._extractTo, member)
                        if osp.exists(path) or osp.isfile(path):
                            continue
                        else:
                            f.extract(member, self._extractTo)
        except Exception as e:
            print(('  catch {}'.format(e)))

    def save(self, r):
        with open(self._filename, 'wb') as f:
            print('  progress ', end='')
            sys.stdout.flush()
            while True:
                buf = r.read(self.BUFSIZE)
                if not buf:
                    break
                f.write(buf)
                print('>', end='')
                sys.stdout.flush()


def GDrive(gid):
    def download_gdrive(dst):
        session = requests.Session()  # re-use cookies

        URL = "https://docs.google.com/uc?export=download"
        response = session.get(URL, params = { 'id' : gid }, stream = True)

        def get_confirm_token(response):  # in case of large files
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    return value
            return None
        token = get_confirm_token(response)

        if token:
            params = { 'id' : gid, 'confirm' : token }
            response = session.get(URL, params = params, stream = True)

        BUFSIZE = 1024 * 1024
        PROGRESS_SIZE = 10 * 1024 * 1024

        sz = 0
        progress_sz = PROGRESS_SIZE
        with open(dst, "wb") as f:
            for chunk in response.iter_content(BUFSIZE):
                if not chunk:
                    continue  # keep-alive

                f.write(chunk)
                sz += len(chunk)
                if sz >= progress_sz:
                    progress_sz += PROGRESS_SIZE
                    print('>', end='')
                    sys.stdout.flush()
        print('')
        return sz
    return download_gdrive

# Data will be downloaded and extracted to ./data by default
data_downloaders = dict(
    face_detection=Downloader(name='face_detection',
        url='https://drive.google.com/u/0/uc?id=1lOAliAIeOv4olM65YDzE55kn6XjiX2l6&export=download',
        sha='0ba67a9cfd60f7fdb65cdb7c55a1ce76c1193df1',
        filename='face_detection.zip'),
    face_recognition=Downloader(name='face_recognition',
        url='https://drive.google.com/u/0/uc?id=1BRIozREIzqkm_aMQ581j93oWoS-6TLST&export=download',
        sha='03892b9036c58d9400255ff73858caeec1f46609',
        filename='face_recognition.zip'),
    facial_expression_recognition=Downloader(name='facial_expression_recognition',
        url='https://drive.google.com/u/0/uc?id=13ZE0Pz302z1AQmBmYGuowkTiEXVLyFFZ&export=download',
        sha='8f757559820c8eaa1b1e0065f9c3bbbd4f49efe2',
        filename='facial_expression_recognition.zip'),
    text=Downloader(name='text',
        url='https://drive.google.com/u/0/uc?id=1lTQdZUau7ujHBqp0P6M1kccnnJgO-dRj&export=download',
        sha='a40cf095ceb77159ddd2a5902f3b4329696dd866',
        filename='text.zip'),
    image_classification=Downloader(name='image_classification',
        url='https://drive.google.com/u/0/uc?id=1qcsrX3CIAGTooB-9fLKYwcvoCuMgjzGU&export=download',
        sha='987546f567f9f11d150eea78951024b55b015401',
        filename='image_classification.zip'),
    human_segmentation=Downloader(name='human_segmentation',
        url='https://drive.google.com/u/0/uc?id=1Kh0qXcAZCEaqwavbUZubhRwrn_8zY7IL&export=download',
        sha='ac0eedfd8568570cad135acccd08a134257314d0',
        filename='human_segmentation.zip'),
    qrcode=Downloader(name='qrcode',
        url='https://drive.google.com/u/0/uc?id=1_OXB7eiCIYO335ewkT6EdAeXyriFlq_H&export=download',
        sha='ac01c098934a353ca1545b5266de8bb4f176d1b3',
        filename='qrcode.zip'),
    object_tracking=Downloader(name='object_tracking',
        url='https://drive.google.com/u/0/uc?id=1_cw5pUmTF-XmQVcQAI8fIp-Ewi2oMYIn&export=download',
        sha='0bdb042632a245270013713bc48ad35e9221f3bb',
        filename='object_tracking.zip'),
    person_reid=Downloader(name='person_reid',
        url='https://drive.google.com/u/0/uc?id=1G8FkfVo5qcuyMkjSs4EA6J5e16SWDGI2&export=download',
        sha='5b741fbf34c1fbcf59cad8f2a65327a5899e66f1',
        filename='person_reid.zip'),
    palm_detection=Downloader(name='palm_detection',
        url='https://drive.google.com/u/0/uc?id=1Z4KvccTZPeZ0qFLZ6saBt_TvcKYyo9JE&export=download',
        sha='4b5bb24a51daab8913957e60245a4eb766c8cf2e',
        filename='palm_detection_20230125.zip'),
    license_plate_detection=Downloader(name='license_plate_detection',
        url='https://drive.google.com/u/0/uc?id=1cf9MEyUqMMy8lLeDGd1any6tM_SsSmny&export=download',
        sha='997acb143ddc4531e6e41365fb7ad4722064564c',
        filename='license_plate_detection.zip'),
    object_detection=Downloader(name='object_detection',
        url='https://drive.google.com/u/0/uc?id=1LUUrQIWYYtiGoNAL_twZvdw5NkC39Swe&export=download',
        sha='4161a5cd3b0be1f51484abacf19dc9a2231e9894',
        filename='object_detection.zip'),
    person_detection=Downloader(name='person_detection',
        url='https://drive.google.com/u/0/uc?id=1RbLyetgqFUTt0IHaVmu6c_b7KeXJgKbc&export=download',
        sha='fbae2fb0a47fe65e316bbd0ec57ba21461967550',
        filename='person_detection.zip'),
)

if __name__ == '__main__':
    selected_data_names = []
    for i in range(1, len(sys.argv)):
        selected_data_names.append(sys.argv[i])
    if not selected_data_names:
        selected_data_names = list(data_downloaders.keys())
    print('Data will be downloaded: {}'.format(str(selected_data_names)))

    download_failed = []
    for selected_data_name in selected_data_names:
        downloader = data_downloaders[selected_data_name]
        if not downloader.get():
            download_failed.append(downloader._name)

    if download_failed:
        print('Data have not been downloaded: {}'.format(str(download_failed)))
