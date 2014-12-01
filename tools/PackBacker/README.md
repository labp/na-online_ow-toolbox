PackBacker
==========

PackBacker is a light tool to download and install 3rd party libraries.



Concept
-------

Job script with one instruction per line:
```
$ cat jobs/example.bb
cxxtest: dest_dir=~;
eigen3: dest_dir=~;
```

Run installation job:
```
$ python3 packbacker.py -j jobs/example.pb
```


Installation & Usage
--------------------

* Clone repository.
```
$ git clone https://github.com/cpieloth/PackBacker.git PackBacker
```
* Create a new branch for your installation jobs (optional).
```
$ cd PackBacker
$ git checkout -b jobs
```
* Create a installation job.
```
$ cp jobs/example.bb jobs/myjob.bb
$ vi jobs/myjob.bb
```
* Commit your setup (optional).
* Run your installation job
```
$ python3 packbacker.py -j jobs/myjob.pb
```
