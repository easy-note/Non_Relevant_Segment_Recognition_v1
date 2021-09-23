import time

class LogHelper():
    """
    logging -> 추후 log library 적용(log 종류에 따라 형식이 약간 다름?)
    log 는 바로바로 write 할 때 마다 작성
    """

    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.log_txt = ""
    
    def writeln(self, log_txt=""):
        if log_txt != "" :
            self.log_txt = self.get_current_time()
        
        self.log_txt += log_txt + '\n'
        self.save()
        self.flush()

    def flush(self):
        self.log_txt = ""

    def get_current_time(self):
        startTime = time.time()
        s_tm = time.localtime(startTime)
        
        return time.strftime('%Y-%m-%d %I:%M:%S %p \t\t', s_tm)

    # save log txt
    def save(self):
        print('=========> SAVING LOG ... | {}'.format(self.save_dir))
        with open(self.save_dir, 'a') as f :
            f.write(self.log_txt)
        