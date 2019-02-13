import os

filenames = os.listdir("./images")
f = open("./labels.txt","a")
count = 0
for filename in filenames:
    value = filename.split(sep='(')
    img_class = value[0]
    f.write(filename)
    f.write(' ')
    f.write(img_class)
    f.write('\n')
    count = count+1
print(count)



