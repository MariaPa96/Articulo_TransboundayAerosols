

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

	def __init__(self,identified=True):

		self.data_folder = 'C:/Users/maria/OneDrive/Documents/Politecnico/Datos_Meteo_Satel/CORAEFES/Articulo/'
		self.lockdown_dates = ['2020-03-20','2020-07-01']
		self.Events = pd.read_csv(self.data_folder+'Datos/EventsExtIdentification_2019-2022_AOD-SO2.csv',index_col=0,parse_dates=True) 

	def Read_Chemi(self,*args,**kwargs):
		## función de lectura de la caracterización química del PM2.5, se tiene en cuenta levoglucosan, metal, iones, cationes
		## fracciones carbonaceas y secondary organic carbon. 

		variables		= kwargs.get('variable',['levo','metal','ion','cation','carbon'])

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

		if 'cation' in variables:
			Cationc=Data.iloc[87:92,1:-10].T
			Cationc.index=pd.to_datetime(Cationc.index)
			Chem['cation']=Cationc.dropna(axis=0,how='all')

		if 'ion' in variables:
			Ionesc=Data.iloc[95:100,1:-10].T
			Ionesc.index=pd.to_datetime(Ionesc.index)
			Chem['ion']=Ionesc.dropna(axis=0,how='all')

		if 'carbon' in variables:
			Carbonc=Data2.iloc[110:129,2:-1].T
			Carbonc.index=pd.to_datetime(Carbonc.index)
			Carbonc=Carbonc.drop(columns=['OC6','OC7','OC8'])
			Chem['carbon'] = Carbonc.dropna(axis=0,how='all').dropna(axis=1,how='all')

		for var in variables:
			Chem[var]=Chem[var][np.logical_not(np.abs((Chem[var]-Chem[var].mean())/Chem[var].astype(float).std())>5)]
	
		## se concatena en un DataFrame el diccionario de los diferentes compuestos químicos.
		self.Chem    = pd.concat(Chem,axis=1)


	def PostHoc_test(self,*args,**kwargs):
	    ## https://pypi.org/project/scikit-posthocs/
		event = kwargs.get('event','omaod')
		variables = kwargs.get('variables',['metal','ion','cation','carbon'])
		filtro = kwargs.get('filtro',True)
		bincompara = kwargs.get('bincompara',1)

		nn=[-10,0,9,20,30]
		self.df_PostHoc={}
		for var in variables:
			Data=self.Chem[var].copy()
			Data['dias']=self.Events['dias_%s'%event][self.Events.index.floor('D').isin(Data.index)]
			if filtro==True:
				Data=Data[(Data.index<self.lockdown_dates[0])|(Data.index>self.lockdown_dates[1])]

			Data['Cat']=np.repeat(np.nan,len(Data))
			for i in range(len(nn)-1):
				Data.loc[lambda df:(df['dias']>=nn[i])&(df['dias']<nn[i+1]),'Cat']=int(i+1)
			Data=Data.dropna(axis=0)
			self.df_PostHoc[var]=pd.DataFrame(columns=self.Chem[var].columns,index=np.arange(1,5))
			for com in Data.columns[:-2]:
				Compara=sp.posthoc_ttest(Data, val_col=com, group_col='Cat', p_adjust='holm').sort_index().round(4)
				self.df_PostHoc[var][com] = Compara.iloc[bincompara,:].sort_index().values


	def plot_SinceEvent(self,*args,**kwargs):
		event = kwargs.get('event','omaod')
		var = kwargs.get('variables','cation')
		filtro = kwargs.get('filtro',True)
		com = kwargs.get('compuesto','Potasio')

		nn=[-10,0,9,20,30]
		Data=self.Chem[var].copy()
		Data['dias']=self.Events['dias_%s'%event][self.Events.index.floor('D').isin(Data.index)]

		if filtro==True:
			Data=Data[(Data.index<self.lockdown_dates[0])|(Data.index>self.lockdown_dates[1])]

		magnitude=pd.DataFrame(index=nn[:-1],columns=['Prom','Median','Q25','Q75','Count'])
		for i in range(len(nn)-1):
			magnitude.loc[nn[i],'Prom']=np.mean(Data[com][(Data['dias']>=nn[i])&(Data['dias']<nn[i+1])])
			magnitude.loc[nn[i],'Median']=np.median(Data[com][(Data[com]>=0)&(Data['dias']>=nn[i])&(Data['dias']<nn[i+1])])
			magnitude.loc[nn[i],'Q25']=np.quantile(Data[com][(Data[com]>=0)&(Data['dias']>=nn[i])&(Data['dias']<nn[i+1])],0.25)
			magnitude.loc[nn[i],'Q75']=np.quantile(Data[com][(Data[com]>=0)&(Data['dias']>=nn[i])&(Data['dias']<nn[i+1])],0.75)
			magnitude.loc[nn[i],'Count']=len(Data[com][(Data['dias']>=nn[i])&(Data['dias']<nn[i+1])])

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

	def pcolor_SinceEvent(self,*args,**kwargs):
		event = kwargs.get('event','omaod')
		variables = kwargs.get('variables',['metal','ion','cation','carbon'])
		filtro = kwargs.get('filtro',True)
		
		ConcEvent = {}
		nn=[-10,0,9,20,30]

		for var in variables:

			Temporal=self.df_PostHoc[var].T
			compuestos=Temporal[Temporal<=0.1].dropna(axis=0,how='all').index
			Data=self.Chem[var][compuestos].copy()
			for column in Data.columns:
				Data[column] = (Data[column] - Data[column].min()) / (Data[column].max() - Data[column].min())   

			Data['dias']=self.Events['dias_%s'%event][self.Events.index.floor('D').isin(Data.index)]		
			ConcEvent[var]= pd.DataFrame(columns=nn[:-1],index=compuestos)

			for com in compuestos:
				for i in range(len(nn)-1):
 					ConcEvent[var].loc[com,nn[i]]=np.mean(Data[com][(Data['dias']>=nn[i])&(Data['dias']<nn[i+1])])
			ConcEvent[var]=ConcEvent[var].astype(float)
		self.ConcEvent = ConcEvent