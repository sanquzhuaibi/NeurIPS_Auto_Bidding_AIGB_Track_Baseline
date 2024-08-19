docker build -t registry.cn-beijing.aliyuncs.com/ocpm_rl/test1:test_5 -f ./Dockerfile .
docker push registry.cn-beijing.aliyuncs.com/ocpm_rl/test1:test_5

#sudo gpasswd -a $USER docker
#newgrp docker