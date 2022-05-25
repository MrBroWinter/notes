# git
## 将已存在的文件夹创建为新的repository
* 全局设置
* git config --global user.name "冬冬"
* git config --global user.email "zdongdong@infervision.com"
1. cd 到指定文件夹
2. git remote remote add origin <ssh地址>   *链接远程仓库*
3. git add .  将该目录下所有文件添加到暂存区*
4. git commit -m "Initial commit  将暂存区内容推送到版本库*
5. git push origin master   *将版本库推送到远程仓库的master分支*

## 分支
git checkout <分支名> # 切换分支
git checkout -b <分支名> # 创建并切换分支

## tag
* git tag *查看所有tag*
1. 在需要打tag的分支上 git tag <tag名>
2. git push origin <tag名>

## 版本回退
git reset --hard 'commit ID'
	
## .gitignore
创建   .gitignore   文件，并添加文件名或目录，add的时候可自动忽略（可用*通配符）

## submodule相关
### submodule的pull操作
1. git clone ssh://git@code.infervision.com:2022/algorithm/algo-bw/engine/heart-cta-engine.git
2. cd heart-cta-engine
3. git checkout <指定分支>
4. git submodule update --init --recursive   *下载该分支对应的submodule版本*
5. git status *查看当前工作区是否干净*
6. poetry install *更新环境（推想）*
7. git fetch origin <目标分支名> *查看目标分支是否最新，若不加目标分支名，则默认当前分支*
8. git pull origin <目标分支名> *将目标分支更新到最新，若不加目标分支名，则默认当前分支*
9. git status *查看当前工作区是否干净*

### submodule的push操作
1. 将submodule中修改的模块checkout一个分支，将修改后的模块push上去
2. git fetch origin *会显示改动后的模块后面跟着（new commits）*
![](file:////tmp/wps-zdongdong/ksohtml/wps1inZyQ.jpg)
3.  最后正常git add/commit/push(最好先创建分支)

## git设置(更好显示版本迭代的情况)
git config --global alias.lg "log --color --graph --branches --pretty=format:'%C(auto)%h %C(auto)%d %C(auto)%s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --decorate --abbrev-commit --all"

## 设置上传url地址
git remote set-url origin <地址>

# 修改为想要设置的远程仓库
git remote set-url origin ssh://git@code.infervision.com:2022/algorithm/alchemy/heart-cta/seg_aortic_dissection2.git