

import numpy as np
import pandas as pd
from glob import glob 
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import xarray as xr
import scikit_posthocs as sp



class Ext:

	def __init__(self,identified=True,data_folder=None,level=1):

		self.data_folder = 'C:/Users/maria/OneDrive/Documents/Politecnico/Datos_Meteo_Satel/CORAEFES/Articulo/' if data_folder==None else data_folder
		#self.lockdown_dates = ['2020-03-20','2020-07-01']
		self.lockdown_dates = ['2020-04-01','2020-06-01']

		self.Events = pd.read_csv(self.data_folder+'Datos/EventsExtIdentification_2019-2022_AOD-SO2%s.csv'%('' if level==1 else '_'+str(int(level*10)).zfill(2)),index_col=0,parse_dates=True) 
		self.Events1 = pd.read_csv(self.data_folder+'Datos/EventsExtIdentification_2019-2022_AOD-SO2.csv',index_col=0,parse_dates=True) 

		self.nn = [-15,-3,4,16] ## 8,26 o 5,16

	def Read_Chemi(self,*args,**kwargs):
		## función de lectura de la caracterización química del PM2.5, se tiene en cuenta levoglucosan, metal, iones, cationes
		## fracciones carbonaceas y secondary organic carbon. 

		variables		= kwargs.get('variable',['levo','metal','ion','cation','carbon','soc'])

		### documento en donde se almacenan los resultados 

		file_chemi = self.data_folder+'Datos/ONUARCAL_CARACTERIZ_ABRIL_2019-OCT_2022_20230213_ARTIC_F-SAT-CARAC1_PM2.5.xlsx'

		## Los resultados estan almacenados en recuedro de diferentes hojas del excel, por eso la seguiente seccion 
		## se encarga de buscar cada tabla para ser leida. 

		Data=pd.read_excel(file_chemi,sheet_name='Carac MED-BEME',index_col=1,skiprows=8)
		Data=Data.dropna(axis=0,how='all').dropna(axis=1,how='all')
		Data1=pd.read_excel(file_chemi,sheet_name='Levoglucosan',index_col=1,skiprows=6,parse_dates=True)
		Data2=pd.read_excel(file_chemi,sheet_name='Facciones Carbono',index_col=2,skiprows=8,parse_dates=True)

		## ya aquí se hace la lectura
		Chem={}
		
		if 'levo' in variables:
			Levoc=Data1.iloc[:-3,1:]
			Chem['levo'] = Levoc.dropna(axis=0)
		if 'metal' in variables:
			Metalc=Data.iloc[58:84,1:-10].T
			Metalc.index=pd.to_datetime(Metalc.index)
			Chem['metal']=Metalc.dropna(axis=0,how='all')#.dropna(axis=0,subset=Metalc.columns[:-1])
			Chem['metal'] = Chem['metal'].rename(columns={'Total':'Metals'})

		if 'cation' in variables:
			Cationc=Data.iloc[87:92,1:-10].T
			Cationc.index=pd.to_datetime(Cationc.index)
			Chem['cation']=Cationc.dropna(axis=0,how='all')

		if 'ion' in variables:
			Ionesc=Data.iloc[95:100,1:-10].T
			Ionesc.index=pd.to_datetime(Ionesc.index)
			Ionesc=Ionesc.rename(columns={'Total':'Anions'})
			Chem['ion']=Ionesc.dropna(axis=0,how='all')

		if 'carbon' in variables:
			Carbonc=Data2.iloc[110:129,2:-1].T
			Carbonc.index=pd.to_datetime(Carbonc.index)
			Carbonc=Carbonc.drop(columns=['OC6','OC7','OC8'])
			Carbonc=Carbonc.rename(columns={'C Total':'C'})
			Chem['carbon'] = Carbonc.dropna(axis=0,how='all').dropna(axis=1,how='all')
		if 'soc' in variables:
			DataSOC = pd.read_excel(self.data_folder+'Datos/ONUARCAL_Calculos_SOC_2019-2022-1.xlsx',
            			 sheet_name='Completo',index_col=1,parse_dates=True)
			SOC = DataSOC.iloc[:247,15:18]
			SOC.index=pd.to_datetime(SOC.index)
			Chem['soc'] = SOC[['SOC','SOC/OC']].dropna(axis=0,how='all').dropna(axis=1,how='all')


		#for var in variables:
		#	Chem[var]=Chem[var][np.logical_not(np.abs((Chem[var]-Chem[var].mean())/Chem[var].astype(float).std())>3)]
	
		## se concatena en un DataFrame el diccionario de los diferentes compuestos químicos.
		self.Chem    = pd.concat(Chem,axis=1)
		self.Crustal = pd.DataFrame(data={'Mg':1.071,'Al':11.43,'Si':29.568,'P':0.297,'S':0.135,'Cl':0.227,'K':0.991,'Ca':2.379,'Ti':1.378,'Mn':0.175,'Fe':10.198,'Cu':0.011,'Zn':0.017},index=['value']).T

	def PostHoc_test(self,*args,**kwargs):
	    ## https://pypi.org/project/scikit-posthocs/
		event = kwargs.get('event','omaod')
		variables = kwargs.get('variables',['metal','ion','cation','carbon','soc'])
		filtro = kwargs.get('filtro',True)
		bincompara = kwargs.get('bincompara',1)

		nn= self.nn.copy()
		self.df_PostHoc={}
		for var in variables:
			Data=self.Chem[var].copy()
			Data['dias']=self.Events['dias_%s'%event][self.Events.index.floor('D').isin(Data.index)]
			
			filtro_event1= 'omaod' if event in ['tcso2','duaod'] else 'tcso2'
			filtro_event2= 'duaod' if event in ['tcso2','omaod'] else 'tcso2'
			Data['dias2']=self.Events1['dias_%s'%filtro_event1][self.Events1.index.floor('D').isin(Data.index)]
			Data['dias3']=self.Events1['dias_%s'%filtro_event2][self.Events1.index.floor('D').isin(Data.index)]
			Data=Data[(Data['dias2']>3)|(Data['dias2']<-3)]	
			Data=Data[(Data['dias3']>3)|(Data['dias3']<-3)]			
			Data=Data.iloc[:,:-2]
			if filtro==True:
				Data=Data[(Data.index<self.lockdown_dates[0])|(Data.index>self.lockdown_dates[1])]

			Data['Cat']=np.repeat(np.nan,len(Data))
			for i in range(len(nn)-1):
				Data.loc[lambda df:(df['dias']>=nn[i])&(df['dias']<nn[i+1]),'Cat']=int(i+1)
			self.Data=Data.dropna(subset='Cat')
			self.df_PostHoc[var]=pd.DataFrame(columns=self.Chem[var].columns,index=np.arange(1,len(nn)))
			for com in self.Data.columns[:-2]:
				if (self.Data[com][self.Data.Cat==2].count()>=10) and (self.Data[com][self.Data.Cat==1].count()>=10) and (self.Data[com][self.Data.Cat==3].count()>=10):
					if (self.Data[com][self.Data.Cat==2].mean()>self.Data[com][self.Data.Cat==1].mean()) and (self.Data[com][self.Data.Cat==2].mean()>self.Data[com][self.Data.Cat==3].mean()):
						Compara=sp.posthoc_mannwhitney(self.Data.dropna(subset=[com])[[com,'Cat']].astype(float), val_col=com, group_col='Cat', p_adjust='holm')#,alternative='less').sort_index().round(4)
						if Compara.loc[1,3]>0.05 and Compara.loc[1,2]<=0.1 and Compara.loc[2,3]<0.2:### >0.1
							##Compara=sp.posthoc_ttest(self.Data, val_col=com, group_col='Cat', p_adjust='holm',equal_var=False).sort_index().round(4)
							self.df_PostHoc[var][com] = Compara.iloc[bincompara,:].sort_index().values


	def Plot_SinceEvent(self,*args,**kwargs):
		event = kwargs.get('event','omaod')
		var = kwargs.get('variables','cation')
		filtro = kwargs.get('filtro',True)
		com = kwargs.get('compuesto','Potasio')

		nn=self.nn.copy()
		Data=self.Chem[var].copy()
		Data['dias']=self.Events['dias_%s'%event][self.Events.index.floor('D').isin(Data.index)]

		filtro_event1= 'omaod' if event in ['tcso2','duaod'] else 'tcso2'
		filtro_event2= 'duaod' if event in ['tcso2','omaod'] else 'tcso2'
		Data['dias2']=self.Events1['dias_%s'%filtro_event1][self.Events1.index.floor('D').isin(Data.index)]
		Data['dias3']=self.Events1['dias_%s'%filtro_event2][self.Events1.index.floor('D').isin(Data.index)]
		Data=Data[(Data['dias2']>3)|(Data['dias2']<-3)]	
		Data=Data[(Data['dias3']>3)|(Data['dias3']<-3)]			
		Data=Data.iloc[:,:-2]

		if filtro==True:
			Data=Data[(Data.index<self.lockdown_dates[0])|(Data.index>self.lockdown_dates[1])]
		self.Data=Data
		magnitude=pd.DataFrame(index=nn[:-1],columns=['Prom','Median','Q25','Q75','Count'])
		for i in range(len(nn)-1):
			magnitude.loc[nn[i],'Prom']=np.mean(Data[com][(Data['dias']>=nn[i])&(Data['dias']<nn[i+1])])
			magnitude.loc[nn[i],'Median']=np.median(Data[com][(Data[com]>=0)&(Data['dias']>=nn[i])&(Data['dias']<nn[i+1])])
			magnitude.loc[nn[i],'Q25']=np.quantile(Data[com][(Data[com]>=0)&(Data['dias']>=nn[i])&(Data['dias']<nn[i+1])],0.25)
			magnitude.loc[nn[i],'Q75']=np.quantile(Data[com][(Data[com]>=0)&(Data['dias']>=nn[i])&(Data['dias']<nn[i+1])],0.75)
			magnitude.loc[nn[i],'Count']=Data[com][(Data['dias']>=nn[i])&(Data['dias']<nn[i+1])].count()

		X=magnitude.index##nn[:-1]+(np.array(nn[1:])-np.array(nn[:-1]))/2

		plt.close()
		fig, ax= plt.subplots(figsize=(4, 4))
		ax.spines['left'].set_color('gray')        # setting up Y-axis tick color to red
		ax.spines['bottom'].set_color('gray')         #setting up above X-axis tick color to red
		ax.spines['right'].set_color('w')        # setting up Y-axis tick color to red
		ax.spines['top'].set_color('w')         #setting up above X-axis tick color to red

		ax.plot(X,magnitude['Prom'].values,marker='o')
		ax.plot(X,magnitude['Median'].values,marker='v',color='c')

		ax.hlines(np.median(Data[com]),-60,270,linestyle='--',color='gray')

		ax.vlines(X,magnitude['Q25'].values,magnitude['Q75'].values)

		for i in range(len(X)):
			ax.hlines(magnitude['Q25'].values[i],X[i]-4,X[i]+4)
			ax.hlines(magnitude['Q75'].values[i],X[i]-4,X[i]+4)
		plt.xlim(-20,X[-1]+10)

		ax.set_ylabel('%s [$\mu g/m^3$]'%com,fontsize=12) 
		ax.set_xlabel('days since a %s event'%event, color = 'k',fontsize=12) 
		self.magnitude = magnitude

	def ConcStd_SinceEvent(self,*args,**kwargs):
		event = kwargs.get('event','omaod')
		variables = kwargs.get('variables',['metal','ion','cation','carbon'])
		filtro = kwargs.get('filtro',True)
		confianza = kwargs.get('confianza',90)

		ConcEvent = {}
		ConcEventQ25 = {}
		ConcEventQ75 = {}

		nn=self.nn.copy()

		for var in variables:
			self.df_PostHoc[var]=self.df_PostHoc[var].dropna(axis=1,how='all')
			Temporal=self.df_PostHoc[var].T
			compuestos=Temporal[Temporal<=1.-(confianza/100.)].dropna(axis=0,how='all').index
			Data=self.Chem[var][compuestos].copy()
			for column in Data.columns:
				Data[column] = (Data[column] - Data[column].mean()) / Data[column].std()   
				#Data[column] = (Data[column] - Data[column].min()) / (Data[column].max() - Data[column].min())   

			Data['dias']=self.Events['dias_%s'%event][self.Events.index.floor('D').isin(Data.index)]		
			ConcEvent[var]= pd.DataFrame(columns=nn[:-1],index=compuestos)
			ConcEventQ25[var]= pd.DataFrame(columns=nn[:-1],index=compuestos)
			ConcEventQ75[var]= pd.DataFrame(columns=nn[:-1],index=compuestos)

			for com in compuestos:
				for i in range(len(nn)-1):
 					ConcEvent[var].loc[com,nn[i]]=np.quantile(Data[com][(Data[com]>=-40)&(Data['dias']>=nn[i])&(Data['dias']<nn[i+1])],0.5)
 					ConcEventQ25[var].loc[com,nn[i]]=np.quantile(Data[com][(Data[com]>=-40)&(Data['dias']>=nn[i])&(Data['dias']<nn[i+1])],0.25)
 					ConcEventQ75[var].loc[com,nn[i]]=np.quantile(Data[com][(Data[com]>=-40)&(Data['dias']>=nn[i])&(Data['dias']<nn[i+1])],0.75)
			ConcEvent[var]=ConcEvent[var].astype(float)
			ConcEventQ25[var]=ConcEventQ25[var].astype(float)
			ConcEventQ75[var]=ConcEventQ75[var].astype(float)
		self.ConcEventMean = ConcEvent
		self.ConcEventQ25 = ConcEventQ25
		self.ConcEventQ75 = ConcEventQ75


	def Whisker_Plot(self,*args,**kwargs):
		event = kwargs.get('event','omaod')
		variables = kwargs.get('variables',['metal','ion','cation','carbon','soc'])
		filtro = kwargs.get('filtro',True)
		confianza = kwargs.get('confianza',90)

		self.PostHoc_test(event=event,filtro=filtro,variables=variables)
		self.ConcStd_SinceEvent(event=event,filtro=filtro,confianza=confianza,variables=variables)
		nn = self.nn

		plt.close()

		fig, ax= plt.subplots(figsize=(5, 10))
		ax.tick_params(axis ='both',labelsize=14)
		plt.xlabel('Standarized concentration [std.]',fontsize=14)

		datos=pd.concat(self.ConcEventMean)
		datos25=pd.concat(self.ConcEventQ25)
		datos75=pd.concat(self.ConcEventQ75)

		Y=np.arange(0,len(pd.concat(self.ConcEventMean))*3,3)

		plt.scatter(datos[nn[1]],Y,color='k')
		plt.hlines(Y,datos25[nn[1]],datos75[nn[1]],color='k',label='IQR event')
		plt.vlines(datos25[nn[1]],Y-0.25,Y+0.25,color='k')
		plt.vlines(datos75[nn[1]],Y-0.25,Y+0.25,color='k')

		plt.scatter(datos[nn[0]],Y+0.5,color='r')
		plt.hlines(Y+0.5,datos25[nn[0]],datos75[nn[0]],color='r',label='IQR before event')
		plt.vlines(datos25[nn[0]],Y+0.25,Y+0.75,color='r')
		plt.vlines(datos75[nn[0]],Y+0.25,Y+0.75,color='r')

		plt.scatter(datos[nn[2]],Y-0.5,color='blue')
		plt.hlines(Y-0.5,datos25[nn[2]],datos75[nn[2]],color='blue',label='IQR after event')
		plt.vlines(datos25[nn[2]],Y-0.75,Y-0.25,color='blue')
		plt.vlines(datos75[nn[2]],Y-0.75,Y-0.25,color='blue')

		plt.vlines(0,-3,Y[-1]+3,linestyle='--',color='grey')
		#plt.invert_yaxis()  # labels read top-to-bottom

		indexlev1 =datos.index.get_level_values(1)
		indexlev0 =datos.index.get_level_values(0)
		replacements = {'Fluoruro':'F$^-$', 'Cloruro':'Cl$^-$', 'Sulfato':'SO$_4^{-2}$'}
		replacer = replacements.get  # For faster gets.
		indexlev=[replacer(n, n) for n in indexlev1 ]

		plt.yticks(Y,indexlev)

		for i in range(len(Y)):
		    value = self.df_PostHoc[indexlev0[i]][indexlev1[i]][1]
		    if (value<=0.1) and (value>0.05):
		        plt.text(datos75[nn[0]][i]+0.05,Y[i]+0.35,'+',color='r')
		        
		    if value<=0.05:
		        plt.text(datos75[nn[0]][i]+0.05,Y[i]+0.35,'*',color='r')
		    value = self.df_PostHoc[indexlev0[i]][indexlev1[i]][3]
		    if (value<=0.1) and (value>0.05):
		        plt.text(datos75[nn[2]][i]+0.05,Y[i]-0.6,'+',color='blue')
		    if value<=0.05:
		        plt.text(datos75[nn[2]][i]+0.05,Y[i]-0.7,'*',color='blue')

		plt.grid(alpha=0.5)
		plt.xlim(-2,2)
		plt.ylim(-3,Y[-1]+3)