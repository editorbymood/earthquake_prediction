"""
Generate Enhanced Earthquake Dataset for New Delhi Region
Run this script to create a comprehensive, realistic earthquake dataset
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.enhanced_data_generator import NewDelhiEarthquakeDataGenerator
from config import DATA_CONFIG, PATHS

def main():
    """Generate enhanced earthquake dataset"""
    
    print("=" * 80)
    print("🌍 ENHANCED EARTHQUAKE DATA GENERATOR FOR NEW DELHI REGION")
    print("=" * 80)
    print(f"🎯 Location Focus: New Delhi & NCR, India")
    print(f"📊 Samples to generate: {DATA_CONFIG['n_samples']}")
    print(f"🎲 Random seed: {DATA_CONFIG['random_state']}")
    print()
    
    # Create data directory
    os.makedirs(PATHS['data_dir'], exist_ok=True)
    
    # Initialize generator
    print("🚀 Initializing Enhanced Data Generator...")
    generator = NewDelhiEarthquakeDataGenerator(
        random_state=DATA_CONFIG['random_state']
    )
    
    # Generate the enhanced dataset
    print("\n" + "=" * 60)
    print("📈 GENERATING ENHANCED EARTHQUAKE DATASET")
    print("=" * 60)
    
    try:
        # Generate base dataset
        enhanced_df = generator.generate_enhanced_dataset(
            n_samples=DATA_CONFIG['n_samples']
        )
        
        print("\n🔧 Adding realistic measurement uncertainties...")
        enhanced_df = generator.add_realistic_noise_and_uncertainties(enhanced_df)
        
        # Save the enhanced dataset
        output_path = os.path.join(PATHS['data_dir'], 'enhanced_earthquake_data.csv')
        enhanced_df.to_csv(output_path, index=False)
        
        print(f"\n✅ Enhanced dataset saved to: {output_path}")
        
        # Generate summary statistics 
        print("\n" + "=" * 60)
        print("📊 DATASET SUMMARY")
        print("=" * 60)
        
        print(f"📍 Geographic Coverage:")
        print(f"   Latitude: {enhanced_df['latitude'].min():.3f}° to {enhanced_df['latitude'].max():.3f}°N")
        print(f"   Longitude: {enhanced_df['longitude'].min():.3f}° to {enhanced_df['longitude'].max():.3f}°E")
        print(f"   Depth: {enhanced_df['depth_km'].min():.1f} to {enhanced_df['depth_km'].max():.1f} km")
        
        print(f"\n🎯 Earthquake Characteristics:")
        print(f"   Magnitude: {enhanced_df['magnitude'].min():.2f} to {enhanced_df['magnitude'].max():.2f}")
        print(f"   Average magnitude: {enhanced_df['magnitude'].mean():.2f} ± {enhanced_df['magnitude'].std():.2f}")
        
        # Magnitude distribution
        mag_bins = [0, 2, 3, 4, 5, 6, 7, 8]
        mag_counts = pd.cut(enhanced_df['magnitude'], bins=mag_bins, right=False).value_counts().sort_index()
        
        print(f"\n🌋 Magnitude Distribution:")
        for interval, count in mag_counts.items():
            percentage = (count / len(enhanced_df)) * 100
            print(f"   {interval}: {count:,} events ({percentage:.1f}%)")
        
        print(f"\n🏗️ Geological Layers:")
        layer_counts = enhanced_df['geological_layer'].value_counts()
        for layer, count in layer_counts.items():
            percentage = (count / len(enhanced_df)) * 100
            print(f"   {layer}: {count:,} events ({percentage:.1f}%)")
        
        print(f"\n⚡ Focal Mechanisms:")
        mechanism_counts = enhanced_df['mechanism_type'].value_counts()
        for mechanism, count in mechanism_counts.items():
            percentage = (count / len(enhanced_df)) * 100
            print(f"   {mechanism.title()}: {count:,} events ({percentage:.1f}%)")
        
        print(f"\n📅 Time Coverage:")
        print(f"   From: {enhanced_df['timestamp'].min()}")
        print(f"   To: {enhanced_df['timestamp'].max()}")
        print(f"   Years covered: {enhanced_df['year'].nunique()}")
        
        print(f"\n📏 Dataset Dimensions:")
        print(f"   Rows: {len(enhanced_df):,}")
        print(f"   Columns: {len(enhanced_df.columns)}")
        print(f"   Features: {len([col for col in enhanced_df.columns if col != 'magnitude'])}")
        print(f"   File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        
        # Feature correlation analysis
        print(f"\n🔗 Top Feature Correlations with Magnitude:")
        numeric_cols = enhanced_df.select_dtypes(include=[np.number]).columns
        correlations = enhanced_df[numeric_cols].corr()['magnitude'].abs().sort_values(ascending=False)
        
        for i, (feature, corr) in enumerate(correlations[1:11].items()):  # Top 10 excluding magnitude itself
            print(f"   {i+1:2d}. {feature:<25} {corr:.3f}")
        
        print(f"\n🎉 SUCCESS! Enhanced dataset generation completed!")
        print(f"📁 Output file: {output_path}")
        print(f"🌟 Ready for earthquake magnitude prediction modeling!")
        
        # Generate a quick preview CSV for inspection
        preview_path = os.path.join(PATHS['data_dir'], 'data_preview.csv')
        enhanced_df.head(100).to_csv(preview_path, index=False)
        print(f"👀 Preview file (first 100 rows): {preview_path}")
        
    except Exception as e:
        print(f"\n❌ Error during dataset generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    
    print("\n" + "=" * 80)
    if success:
        print("🏆 ENHANCED DATA GENERATION COMPLETED SUCCESSFULLY!")
        print("\n💡 Next steps:")
        print("   1. Run the main earthquake prediction pipeline")
        print("   2. The enhanced data will be automatically used")
        print("   3. Enjoy improved model accuracy with realistic data!")
    else:
        print("💥 DATA GENERATION FAILED!")
        print("\n🔧 Troubleshooting:")
        print("   1. Check error messages above")
        print("   2. Ensure all dependencies are installed")
        print("   3. Verify write permissions in data directory")
    
    print("=" * 80)
