import numpy as np 
np.random.seed(42)
age=np.random.randint(20,81, 100)
bp=np.random.randint(90,181,100)
col=np.random.randint(150,301,100)
age=age.reshape(100,1)
bp=bp.reshape(100,1)
col=col.reshape(100,1)
patients=np.concatenate([age,bp,col],axis=1)
sick2 = np.zeros(30, dtype=int)
sick1 = np.ones(70, dtype=int)
disease_labels = np.concatenate([sick1, sick2])
np.random.shuffle(disease_labels)
print("the average age=",np.mean(patients,axis=0)[0]," \n the average bp=",np.mean(patients,axis=0)[1],"\n the average cholesterol=",np.mean(patients,axis=0)[2])
print("the std age=",np.std(patients,axis=0)[0]," \n the std bp=",np.std(patients,axis=0)[1],"\n the std cholesterol=",np.std(patients,axis=0)[2])
print("the youngest age is=", np.min(patients, axis=0)[0])
print("the olddestt age is=", np.max(patients, axis=0)[0])
print("the highest bp is=", np.max(patients, axis=0)[1])
print("the lowest bp is=", np.min(patients, axis=0)[1])
disease_labels = disease_labels.reshape(-1, 1)
patients = np.concatenate([patients, disease_labels], axis=1)
average=0
ages=0
for x in patients:
    if x[3]==1:
        average= average+1
        ages=ages+x[0]

print("the average age of sick patients is =",ages/average)
chol=0
average=0
for x in patients:
    if x[3]==1:
        average= average+1
        chol=chol+x[2]
print("the average cholesterol of sick patients is =",chol/average)
average=0
bpp=0
for x in patients:
    if x[3]==0:
        average= average+1
        bpp=bpp+x[1]
print("the average bp of healthy patients is =",bpp/average)
count=0
for x in patients: 
    if x[0]==60:
        count =count+1
print("Patients over 60:",count)

count=0
for x in patients: 
    if x[1]>140:
        count =count+1
print("Patients with high bp are :",count)

count=0
for x in patients: 
    if x[2]>240:
        count =count+1
print("Patients with high cholesterol are :",count)

count=0
for x in patients: 
    if x[1]>140 and x[0]==60:
        count =count+1
print("Patients over 60 AND with high bp:",count)

count=0
for x in patients: 
    if x[0]<40 and x[3]==1:
        count =count+1
print("Patients who are young (<40) but have disease",count)

alert_list=[]
count=0
for x in patients:
    count=count+1
    if x[0] > 65 and x[1]> 150 and x[2] > 250 :

        y=list(x)
        y.append(count)
        alert_list.append(y)
   
 
 
#adding bmi colomn
bmi=np.random.randint(18,36,100)
bmi=bmi.reshape(-1,1)
patients=np.concatenate([patients,bmi],axis=1)
new_matrix=[]
for x in patients:
    y,z,_,_,t=x
    sis=[y,z,t]
    new_matrix.append(sis)
new_matrice=np.array(new_matrix)
#reshaping
t_new_matrice=new_matrice.T
print(t_new_matrice)
#finding poeple in danger zone:
for x in range(0,100):
    age1 = t_new_matrice[0, x]
    bp1 = t_new_matrice[1, x]
    bmi1 = t_new_matrice[2, x]
    
    if (50 < age1 < 70) or (140 < bp1 < 160) or (220 < bmi1 < 280):
        print(f"Patient number {x+1} is in danger")





