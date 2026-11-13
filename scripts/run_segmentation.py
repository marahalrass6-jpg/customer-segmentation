from __future__ import annotations
import json, sys
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); OUT=ROOT/'artifacts'; OUT.mkdir(exist_ok=True)

def load_transactions(raw_dir:Path)->pd.DataFrame:
    files=sorted(raw_dir.rglob('*.xlsx'))
    if not files:
        csv_files=sorted(raw_dir.rglob('*.csv'))
        if csv_files:
            df=pd.read_csv(csv_files[0]); df['invoicedate']=pd.to_datetime(df['invoicedate'],errors='coerce'); return df
        raise FileNotFoundError('No workbook or cleaned CSV found; run scripts/download_data.py first.')
    frames=[]
    for file in files:
        book=pd.ExcelFile(file)
        for sheet in book.sheet_names: frames.append(pd.read_excel(file,sheet_name=sheet))
    df=pd.concat(frames,ignore_index=True); df.columns=[str(c).strip().lower().replace(' ','_') for c in df.columns]
    df=df.rename(columns={'invoice':'invoice_no','price':'unit_price','customer_id':'customer_id'})
    df['invoice_no']=df['invoice_no'].astype(str).str.strip(); df['quantity']=pd.to_numeric(df['quantity'],errors='coerce'); df['unit_price']=pd.to_numeric(df['unit_price'],errors='coerce'); df['customer_id']=pd.to_numeric(df['customer_id'],errors='coerce'); df['invoicedate']=pd.to_datetime(df['invoicedate'],errors='coerce')
    df=df[~df['invoice_no'].str.upper().str.startswith('C')]; df=df[(df['quantity']>0)&(df['unit_price']>0)].dropna(subset=['customer_id','invoicedate']); df['revenue']=df['quantity']*df['unit_price']; return df

def make_rfm(df):
    snapshot=df['invoicedate'].max().normalize()+pd.Timedelta(days=1)
    rfm=df.groupby('customer_id').agg(last_purchase=('invoicedate','max'),frequency=('invoice_no','nunique'),monetary_value=('revenue','sum')).reset_index(); rfm['recency_days']=(snapshot-rfm['last_purchase'].dt.normalize()).dt.days; rfm=rfm.drop(columns='last_purchase'); return rfm

def segment_name(row,med):
    r,f,m=row['recency_days'],row['frequency'],row['monetary_value']
    if r<=med['recency_days'] and f>=med['frequency'] and m>=med['monetary_value']: return 'High-Value Loyal'
    if r>med['recency_days'] and f<med['frequency']: return 'At-Risk'
    if f>=med['frequency'] and m<med['monetary_value']: return 'Frequent Low-Value'
    if r<=med['recency_days']: return 'Recent Potential'
    return 'Occasional / Low-Value'

def main():
    df=load_transactions(ROOT/'data'/'raw'); rfm=make_rfm(df); features=['recency_days','frequency','monetary_value']; matrix=np.log1p(rfm[features]); scaler=StandardScaler(); z=scaler.fit_transform(matrix)
    rows=[]
    for k in range(2,9):
        km=KMeans(n_clusters=k,n_init=20,random_state=42); labels=km.fit_predict(z); rows.append({'k':k,'silhouette_score':float(silhouette_score(z,labels)),'inertia':float(km.inertia_)})
    selection=pd.DataFrame(rows); selection.to_csv(OUT/'cluster_selection.csv',index=False); best_k=int(selection.loc[selection.silhouette_score.idxmax(),'k']); km=KMeans(n_clusters=best_k,n_init=20,random_state=42); rfm['cluster']=km.fit_predict(z)
    profiles=rfm.groupby('cluster')[features].agg(['count','mean','median']).round(2); profiles.columns=['_'.join(c) for c in profiles.columns]; profiles=profiles.reset_index(); med=rfm[features].median(); simple=rfm.groupby('cluster')[features].mean().reset_index(); simple['segment_name']=simple.apply(lambda x:segment_name(x,med),axis=1); rfm=rfm.merge(simple[['cluster','segment_name']],on='cluster'); profiles=profiles.merge(simple[['cluster','segment_name']],on='cluster'); profiles.to_csv(OUT/'segment_profiles.csv',index=False); rfm.to_csv(OUT/'customer_segments.csv',index=False)
    recs={'High-Value Loyal':'VIP retention, early access, and cross-sell offers','At-Risk':'Win-back campaign, personalized reminder, and limited incentive','Frequent Low-Value':'Bundle offers and minimum-spend promotions','Recent Potential':'Second-purchase journey and product education','Occasional / Low-Value':'Low-cost reactivation and newsletter testing'}; pd.DataFrame([{'segment_name':s,'recommended_action':recs.get(s,'Test targeted campaign')} for s in sorted(rfm.segment_name.unique())]).to_csv(OUT/'segment_recommendations.csv',index=False)
    sns.set_theme(style='whitegrid'); plt.figure(figsize=(9,5)); sns.lineplot(data=selection,x='k',y='inertia',marker='o',label='Inertia'); ax2=plt.twinx(); sns.lineplot(data=selection,x='k',y='silhouette_score',marker='s',color='darkorange',label='Silhouette',ax=ax2); plt.title('Cluster Selection'); plt.tight_layout(); plt.savefig(OUT/'elbow_silhouette.png',dpi=160); plt.close()
    plt.figure(figsize=(10,6)); sns.scatterplot(data=rfm,x='recency_days',y='monetary_value',hue='segment_name',size='frequency',sizes=(20,220),alpha=.75); plt.yscale('log'); plt.title('Customer Segments: Recency vs Monetary Value'); plt.tight_layout(); plt.savefig(OUT/'segment_scatter.png',dpi=160); plt.close()
    scaled=pd.DataFrame(z,columns=features); scaled['segment_name']=rfm.segment_name; heat=scaled.groupby('segment_name')[features].mean(); plt.figure(figsize=(9,5)); sns.heatmap(heat,cmap='vlag',center=0,annot=True,fmt='.2f'); plt.title('Standardized RFM Segment Profiles'); plt.tight_layout(); plt.savefig(OUT/'segment_profile_heatmap.png',dpi=160); plt.close()
    fig,axes=plt.subplots(1,3,figsize=(16,5));
    for col,ax in zip(features,axes): sns.histplot(data=rfm,x=col,hue='segment_name',element='step',stat='density',common_norm=False,ax=ax); ax.set_title(col)
    plt.tight_layout(); plt.savefig(OUT/'rfm_distribution.png',dpi=160); plt.close()
    joblib.dump({'scaler':scaler,'model':km,'features':features,'best_k':best_k},OUT/'customer_segmentation.joblib'); report={'transactions_after_cleaning':len(df),'customers':len(rfm),'candidate_k_range':[2,8],'selected_k':best_k,'selected_silhouette_score':float(selection.loc[selection.k==best_k,'silhouette_score'].iloc[0]),'segments':profiles.to_dict(orient='records')}; (OUT/'segmentation_report.json').write_text(json.dumps(report,indent=2,default=str)); print(json.dumps(report,indent=2,default=str))
if __name__=='__main__': main()
