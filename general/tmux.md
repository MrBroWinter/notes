# tmux

* 作用：ssh断开连接后，保持后台进程继续运行

* 安装：sudo apt-get install tmux

* 启动tmux：tmux new -s <session-name>

* 回话分离：(在终端窗口中输入)tmux detach或在tmux窗口先快捷键Ctrl+b 然后按d

* 查看tmux：在tmux窗口外：tmux ls

* 接入会话：tmux attach -t <session-name>

* 杀死会话： tmux kill-session -t <session-name>

* 切换会话：tmux switch -t <session-name>

* 新建窗口：tmux new-window -n <window-name>

* 切换窗口：ctrl + b n 快速切换到下一个窗口

# 快捷键
* Ctrl+b d：分离当前会话。

* Ctrl+b s：列出所有会话。

* Ctrl+b $：重命名当前会话
	
# 划分窗格

* tmux split-window (上下划分)

* tmux split-window -h (左右划分)
	
# 切换窗口
	
* Ctrl + b <然后按方向键>