
# 读取summary的东西
import tensorflow as tf
import os


def read_event(event_name):
    # e，即event，代表某一个batch的日志记录,读取日志记录
    for e in tf.train.summary_iterator(event_name):
        # v，即value，代表这个batch的某个已记录的观测值，loss或者accuracy
        for v in e.summary.value:
            # if v.tag == 'loss' or v.tag == 'accuracy':
            #     print(v.simple_value)
            print([v.tag, v.simple_value])

    from tensorboard.backend.event_processing import event_accumulator
    # 加载日志数据
    ea = event_accumulator.EventAccumulator(event_name)
    ea.Reload()
    print(ea.scalars.Keys())

    loss = ea.scalars.Items('losses/ent_loss')
    print(len(loss))
    print([(i.step, i.value) for i in loss])

    import matplotlib.pyplot as plt
    step = [i.step for i in loss]
    loss_value = [i.value for i in loss]
    plt.plot(step, loss_value)
    plt.savefig("loss.png")

    # fig=plt.figure(figsize=(6,4))
    # ax1=fig.add_subplot(111)
    # val_acc=ea.scalars.Items('val_acc')
    # ax1.plot([i.step for i in val_acc],[i.value for i in val_acc],label='val_acc')
    # ax1.set_xlim(0)
    # acc=ea.scalars.Items('acc')
    # ax1.plot([i.step for i in acc],[i.value for i in acc],label='acc')
    # ax1.set_xlabel("step")
    # ax1.set_ylabel("")
    #
    # plt.legend(loc='lower right')
    # plt.show()


if __name__ == "__main__":
    # train_dir = '../model/train/sent_word_rem/transformer/'
    # file_list = os.listdir(train_dir)
    # event_name = None
    # for file in file_list:
    #     if 'event' in file:
    #         event_name = os.path.join(train_dir, file)
    #         break
    # if event_name:
    #     print("the event file is %s" % event_name)
    # else:
    #     raise ValueError("not found the event file")
    event_name = "events.out.tfevents.1587810220.iZm5egodyh41n9y8wu2ulnZ"
    read_event(event_name)
