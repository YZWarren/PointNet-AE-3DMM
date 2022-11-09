SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

cd $SCRIPTPATH/../data/raw
wget http://modelnet.cs.princeton.edu/ModelNet40.zip
unzip ModelNet40.zip
rm ModelNet40.zip
mv ModelNet40 modelnet40
cd -