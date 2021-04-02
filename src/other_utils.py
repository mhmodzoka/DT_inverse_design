#%% SAVING ========================================================================================
def df_to_csv(df_in,filename, scaling_factors):
    """Undo the scaling effects and save the DataFrame to CSV"""
    df = df_in.copy()
        
    if 'Area/Vol' in df.columns:
        df['Area/Vol'] /= scaling_factors['Area/Volume']
    
    if 'log Area/Vol' in df.columns:
        AV_scaled = np.exp(df['log Area/Vol']); print(AV_scaled)
        AV = AV_scaled / scaling_factors['Area/Volume']; print(AV_scaled)
        df['log Area/Vol'] = np.log(AV)
        
        if not('Area/Vol' in df.columns):
            df['Area/Vol'] = np.exp(df['log Area/Vol'])
        
    if 'ShortestDim' in df.columns:
        df['ShortestDim'] /= scaling_factors['Length']
        
    if 'MiddleDim' in df.columns:
        df['MiddleDim'] /= scaling_factors['Length']
        
    if 'LongDim' in df.columns:
        df['LongDim'] /= scaling_factors['Length']
        
    df.to_csv(path_or_buf=filename)