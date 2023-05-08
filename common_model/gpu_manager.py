import os
import logging


class GPUManager:
    """
    qargs:
        query arguments
    A manager which can list all available GPU devices
    and sort them and choice the most free one.Unspecified
    ones pref.
    GPU设备管理器，考虑列举出所有可用GPU设备，并加以排序，自动选出
    最空闲的设备。在一个GPUManager对象内会记录每个GPU是否已被指定，
    优先选择未指定的GPU。
    from QuantumLiu
    """

    def __init__(self, qargs=[], logger=logging):
        """ """
        self.qargs = qargs
        self.gpus = GPUManager.query_gpu(qargs)
        for gpu in self.gpus:
            gpu["specified"] = False
        self.gpu_num = len(self.gpus)
        self.logger = logger

    @staticmethod
    def parse(line, qargs):
        """
        line:
            a line of text
        qargs:
            query arguments
        return:
            a dict of gpu infos
        Pasing a line of csv format text returned by nvidia-smi
        解析一行nvidia-smi返回的csv格式文本
        """
        numberic_args = [
            "memory.free",
            "memory.total",
            "power.draw",
            "power.limit",
        ]  # 可计数的参数
        power_manage_enable = lambda v: (
            not "Not Support" in v
        )  # lambda表达式，显卡是否滋瓷power management（笔记本可能不滋瓷）
        to_numberic = lambda v: float(
            v.upper().strip().replace("MIB", "").replace("W", "")
        )  # 带单位字符串去掉单位
        process = lambda k, v: (
            (int(to_numberic(v)) if power_manage_enable(v) else 1)
            if k in numberic_args
            else v.strip()
        )
        return {k: process(k, v) for k, v in zip(qargs, line.strip().split(","))}

    @staticmethod
    def query_gpu(qargs=[]):
        """
        qargs:
            query arguments
        return:
            a list of dict
        Querying GPUs infos
        查询GPU信息
        """
        qargs = [
            "index",
            "gpu_name",
            "memory.free",
            "memory.total",
            "power.draw",
            "power.limit",
        ] + qargs
        cmd = "nvidia-smi --query-gpu={} --format=csv,noheader".format(",".join(qargs))
        results = os.popen(cmd).readlines()
        return [GPUManager.parse(line, qargs) for line in results]

    @staticmethod
    def by_power(d):
        """
        helper function fo sorting gpus by power
        """
        power_infos = (d["power.draw"], d["power.limit"])
        if any(v == 1 for v in power_infos):
            print("Power management unable for GPU {}".format(d["index"]))
            return 1
        return float(d["power.draw"]) / d["power.limit"]

    def _sort_by_memory(self, gpus, by_size=False):
        if by_size:
            self.logger.info("Sorted by free memory size")
            return sorted(gpus, key=lambda d: d["memory.free"], reverse=True)
        else:
            self.logger.info("Sorted by free memory rate")
            return sorted(
                gpus,
                key=lambda d: float(d["memory.free"]) / d["memory.total"],
                reverse=True,
            )

    def _sort_by_power(self, gpus):
        return sorted(gpus, key=self.by_power)

    def _sort_by_custom(self, gpus, key, reverse=False, qargs=[]):
        if isinstance(key, str) and (key in qargs):
            return sorted(gpus, key=lambda d: d[key], reverse=reverse)
        if isinstance(key, type(lambda a: a)):
            return sorted(gpus, key=key, reverse=reverse)
        raise ValueError(
            "The argument 'key' must be a function or a key in query args,please read the documention of nvidia-smi"
        )

    def auto_choice(self, mode=0):
        """
        mode:
            0:(default)sorted by free memory size
        return:
            a TF device object
        Auto choice the freest GPU device,not specified
        ones
        自动选择最空闲GPU,返回索引
        """
        for old_infos, new_infos in zip(self.gpus, self.query_gpu(self.qargs)):
            old_infos.update(new_infos)
        unspecified_gpus = [
            gpu for gpu in self.gpus if not gpu["specified"]
        ] or self.gpus

        if mode == 0:
            self.logger.info("Choosing the GPU device has largest free memory...")
            chosen_gpu = self._sort_by_memory(unspecified_gpus, True)[0]
        elif mode == 1:
            self.logger.info("Choosing the GPU device has highest free memory rate...")
            chosen_gpu = self._sort_by_power(unspecified_gpus)[0]
        elif mode == 2:
            self.logger.info("Choosing the GPU device by power...")
            chosen_gpu = self._sort_by_power(unspecified_gpus)[0]
        else:
            self.logger.info("Given an unaviliable mode,will be chosen by memory")
            chosen_gpu = self._sort_by_memory(unspecified_gpus)[0]
        chosen_gpu["specified"] = True
        index = chosen_gpu["index"]
        self.logger.info(
            "Using GPU {i}:\n{info}".format(
                i=index,
                info="\n".join([str(k) + ":" + str(v) for k, v in chosen_gpu.items()]),
            )
        )
        return int(index)


if __name__ == "__main__":

    gm = GPUManager()
    gpu_index = gm.auto_choice()
    print("gpu_index", gpu_index)
