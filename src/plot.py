titles = ['Training Loss','Validation Loss','Training Accuracy', 'Validation Accuracy', 'Validation Top 5 Accuracy']
suffixes = ['loss','val_loss','acc','val_acc','val_top_k']

for title, suffix in zip(titles, suffixes):
    fig, ax = plt.subplots()
    ############ RENAME PLOT TITLES AND NAME SUFFIX FOR TYPE OF FILE
    plt.title(title)

    # prefix = ['20171118', '20171120-RN50Real', '20171120-RN50Reg', '20171121-RN50']
    prefix = ['20171120-RN50Real', '20171121-RN50', '20171120-RN50Reg']
    labels = ['ResNet50 run 1', 'ResNet50 run 2', 'ResNet50 reg']

    for pref, label in zip(prefix, labels):
        y = readfile('results/'+pref+'-'+suffix)
        print('reading file:', pref+suffix)
        x = []
        for i in range(len(y)):
            x.append(i+1)
        plt.plot(x, y, '-o', label=label)

    plt.xticks(range(0,21,2))
    ax.set_xlabel('epochs')
    ax.set_ylabel('percentages')
    plt.legend(bbox_to_anchor=(.5, .3), loc=2, borderaxespad=0.)

    plt.savefig(suffix + '.pdf')

    plt.show()
