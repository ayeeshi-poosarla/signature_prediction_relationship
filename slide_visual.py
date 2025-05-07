import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, Polygon
import numpy as np

def create_clinical_relevance_diagram():
    """Create a clinical decision tree visualization showing practical applications."""
    
    # Create figure
    plt.figure(figsize=(16, 10))
    
    # Set up the drawing area
    plt.axis('off')
    plt.xlim(0, 12)
    plt.ylim(0, 8)
    
    # Add title
    plt.text(6, 7.6, "Clinical Applications of Mutation Signature Analysis", 
             fontsize=22, fontweight='bold', ha='center')
    
    # --- LEVEL 1: PATIENT SAMPLE ---
    
    # Draw patient sample box
    patient_sample = Rectangle((5, 6.5), 2, 0.8, facecolor='lightblue', edgecolor='black', 
                               linewidth=2, alpha=0.8, zorder=2)
    plt.gca().add_patch(patient_sample)
    plt.text(6, 6.9, "Patient Tumor Sample", fontsize=14, fontweight='bold', ha='center')
    
    # Add small DNA icon in the box
    dna_x = np.linspace(5.3, 5.7, 20)
    dna_y1 = 0.05*np.sin(8*np.pi*dna_x) + 6.7
    dna_y2 = 0.05*np.sin(8*np.pi*dna_x + np.pi) + 6.7
    plt.plot(dna_x, dna_y1, 'blue', linewidth=2, alpha=0.7)
    plt.plot(dna_x, dna_y2, 'blue', linewidth=2, alpha=0.7)
    
    # Connect lines between strands
    for i in range(0, len(dna_x), 3):
        plt.plot([dna_x[i], dna_x[i]], [dna_y1[i], dna_y2[i]], 'blue', linewidth=1, alpha=0.7)
        
    # --- LEVEL 2: MUTATION ANALYSIS ---
    
    # Draw arrow down to analysis
    plt.arrow(6, 6.5, 0, -0.4, head_width=0.1, head_length=0.1, 
              fc='black', ec='black', zorder=1)
    
    # Draw analysis box
    analysis_box = Rectangle((4.5, 5.5), 3, 0.6, facecolor='lightgreen', edgecolor='black',
                             linewidth=2, alpha=0.8, zorder=2)
    plt.gca().add_patch(analysis_box)
    plt.text(6, 5.8, "Mutation Signature Analysis", fontsize=14, fontweight='bold', ha='center')
    
    # Add mutation signature icons
    plt.text(5.0, 5.65, "C>A", fontsize=10, color='red', fontweight='bold')
    plt.text(5.5, 5.65, "TMB", fontsize=10, color='blue', fontweight='bold')
    plt.text(6.0, 5.65, "SNVs", fontsize=10, color='purple', fontweight='bold')
    plt.text(6.7, 5.65, "Types", fontsize=10, color='green', fontweight='bold')
    
    # --- LEVEL 3: BRANCHING PATHS ---
    
    # Draw left branch
    plt.arrow(6, 5.5, -2, -0.8, head_width=0.1, head_length=0.1, 
              fc='black', ec='black', zorder=1)
    
    # Draw right branch
    plt.arrow(6, 5.5, 2, -0.8, head_width=0.1, head_length=0.1, 
              fc='black', ec='black', zorder=1)
    
    # Draw left (cold) tumor profile
    cold_profile = Rectangle((2.5, 4.2), 3, 0.6, facecolor='lightcoral', edgecolor='black',
                           linewidth=2, alpha=0.8, zorder=2)
    plt.gca().add_patch(cold_profile)
    plt.text(4, 4.5, "Cold Tumor Profile", fontsize=14, fontweight='bold', ha='center')
    
    # Add cold tumor characteristics
    plt.text(2.7, 4.3, "• Low C>A frequency", fontsize=10)
    plt.text(4.3, 4.3, "• Low TMB", fontsize=10)
    
    # Draw right (hot) tumor profile
    hot_profile = Rectangle((6.5, 4.2), 3, 0.6, facecolor='lightgreen', edgecolor='black',
                          linewidth=2, alpha=0.8, zorder=2)
    plt.gca().add_patch(hot_profile)
    plt.text(8, 4.5, "Hot Tumor Profile", fontsize=14, fontweight='bold', ha='center')
    
    # Add hot tumor characteristics
    plt.text(6.7, 4.3, "• High C>A frequency", fontsize=10)
    plt.text(8.3, 4.3, "• High TMB", fontsize=10)
    
    # --- LEVEL 4: BRANCHING TREATMENTS ---
    
    # Draw arrows from cold profile to treatments
    plt.arrow(3, 4.2, -1, -0.8, head_width=0.1, head_length=0.1, fc='black', ec='black', zorder=1)
    plt.arrow(4, 4.2, 0, -0.8, head_width=0.1, head_length=0.1, fc='black', ec='black', zorder=1)
    plt.arrow(5, 4.2, 1, -0.8, head_width=0.1, head_length=0.1, fc='black', ec='black', zorder=1)
    
    # Draw arrows from hot profile to treatments
    plt.arrow(7, 4.2, -1, -0.8, head_width=0.1, head_length=0.1, fc='black', ec='black', zorder=1)
    plt.arrow(8, 4.2, 0, -0.8, head_width=0.1, head_length=0.1, fc='black', ec='black', zorder=1)
    plt.arrow(9, 4.2, 1, -0.8, head_width=0.1, head_length=0.1, fc='black', ec='black', zorder=1)
    
    # Treatment boxes for cold tumors
    treatments_cold = [
        {"x": 1.5, "y": 3.0, "label": "Targeted\nTherapy", "color": "lightyellow"},
        {"x": 4.0, "y": 3.0, "label": "Standard\nChemotherapy", "color": "lightyellow"},
        {"x": 6.0, "y": 3.0, "label": "Combination\nTherapy", "color": "lightyellow"}
    ]
    
    # Treatment boxes for hot tumors
    treatments_hot = [
        {"x": 6.0, "y": 3.0, "label": "Combination\nTherapy", "color": "lightyellow"}, 
        {"x": 8.0, "y": 3.0, "label": "Immune\nCheckpoint\nBlockade", "color": "lightyellow"},
        {"x": 10.0, "y": 3.0, "label": "Personalized\nVaccines", "color": "lightyellow"}
    ]
    
    # Create all treatment boxes
    treatments = treatments_cold + treatments_hot
    for t in treatments:
        box = Rectangle((t["x"]-0.75, t["y"]-0.3), 1.5, 0.6, facecolor=t["color"], 
                        edgecolor='black', linewidth=1.5, alpha=0.8, zorder=2)
        plt.gca().add_patch(box)
        plt.text(t["x"], t["y"], t["label"], fontsize=10, fontweight='bold', ha='center', va='center')
    
    # --- LEVEL 5: RESPONSE RATES ---
    
    # Add response probabilities for cold tumor treatments
    plt.text(1.5, 2.5, "Response: 25%", fontsize=11, color='darkred', ha='center', fontweight='bold')
    plt.text(4.0, 2.5, "Response: 20%", fontsize=11, color='darkred', ha='center', fontweight='bold')
    plt.text(6.0, 2.5, "Response: 35%", fontsize=11, color='darkblue', ha='center', fontweight='bold')
    
    # Add response probabilities for hot tumor treatments
    plt.text(6.0, 2.5, "Response: 35%", fontsize=11, color='darkblue', ha='center', fontweight='bold')
    plt.text(8.0, 2.5, "Response: 45%", fontsize=11, color='darkgreen', ha='center', fontweight='bold')
    plt.text(10.0, 2.5, "Response: 55%", fontsize=11, color='darkgreen', ha='center', fontweight='bold')
    
    # --- ADDITIONAL INFORMATION BOXES ---
    
    # Add model performance box
    performance_box = Rectangle((1.0, 1.4), 4, 0.8, facecolor='lavender', 
                               edgecolor='black', linewidth=1.5, alpha=0.7)
    plt.gca().add_patch(performance_box)
    plt.text(3.0, 1.9, "Predictive Model Performance", fontsize=12, fontweight='bold', ha='center')
    plt.text(1.2, 1.7, "• Accuracy: 67.5%", fontsize=11)
    plt.text(1.2, 1.5, "• ROC-AUC: 0.718", fontsize=11)
    plt.text(3.5, 1.7, "• CV Accuracy: 65.8%", fontsize=11)
    plt.text(3.5, 1.5, "• Comparable to clinical biomarkers", fontsize=11)
    
    # Add clinical applications box
    applications_box = Rectangle((7.0, 1.4), 4, 0.8, facecolor='lavender', 
                                edgecolor='black', linewidth=1.5, alpha=0.7)
    plt.gca().add_patch(applications_box)
    plt.text(9.0, 1.9, "Potential Clinical Applications", fontsize=12, fontweight='bold', ha='center')
    plt.text(7.2, 1.7, "• Personalized immunotherapy selection", fontsize=11)
    plt.text(7.2, 1.5, "• Patient stratification for clinical trials", fontsize=11)
    plt.text(7.2, 1.3, "• Combination therapy guidance", fontsize=11)
    
    # --- ICONS FOR TREATMENTS ---
    
    # Targeted therapy icon (pill)
    pill = Polygon([(1.5-0.2, 3.3), (1.5+0.2, 3.3), (1.5+0.2, 3.2), (1.5-0.2, 3.2)], 
                  facecolor='orange', edgecolor='black', alpha=0.8, zorder=3)
    plt.gca().add_patch(pill)
    
    # Chemotherapy icon (IV bag)
    iv_top = Rectangle((4-0.1, 3.3), 0.2, 0.1, facecolor='white', edgecolor='black', alpha=0.8, zorder=3)
    plt.gca().add_patch(iv_top)
    plt.plot([4], [3.3], marker='v', color='blue', markersize=8)
    
    # Checkpoint inhibitor icon (T cell and receptor)
    t_cell = Circle((8, 3.3), 0.1, facecolor='skyblue', edgecolor='black', alpha=0.8, zorder=3)
    plt.gca().add_patch(t_cell)
    plt.text(8, 3.3, "T", ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Personalized vaccine icon (syringe)
    plt.plot([10-0.1, 10+0.1], [3.3, 3.35], 'k-', linewidth=2)
    plt.plot([10+0.1], [3.35], marker='>', color='black', markersize=6)
    
    # --- LEGEND AND NOTES ---
    
    # Add explanatory notes
    plt.figtext(0.5, 0.05, 
                "Note: Response rates are hypothetical and based on projected outcomes. Actual rates would depend on cancer type and other factors.",
                fontsize=10, style='italic', ha='center')
    
    # Add legend for mutation signatures
    legend_box = Rectangle((2, 7), 7, 0.4, facecolor='white', edgecolor='black', alpha=0.5)
    plt.gca().add_patch(legend_box)
    plt.text(2.3, 7.2, "Legend:", fontsize=10, fontweight='bold')
    plt.text(3.0, 7.2, "C>A", fontsize=10, color='red', fontweight='bold')
    plt.text(3.5, 7.2, "= C>A transversions", fontsize=10)
    plt.text(5.5, 7.2, "TMB", fontsize=10, color='blue', fontweight='bold')
    plt.text(6.0, 7.2, "= Tumor Mutation Burden", fontsize=10)
    plt.text(8.2, 7.2, "Response rates color-coded by efficacy", fontsize=10)

    plt.tight_layout(rect=[0, 0.07, 1, 0.97])
    plt.savefig('slide9_clinical_relevance.png', dpi=300, bbox_inches='tight')
    plt.close()

# Call the function to create the visualization
if __name__ == "__main__":
    create_clinical_relevance_diagram()
    print("Clinical relevance diagram created: slide9_clinical_relevance.png")