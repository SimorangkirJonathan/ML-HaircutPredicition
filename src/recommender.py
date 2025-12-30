"""
Hairstyle recommender based on face shape and hair type combination.
"""


class HairstyleRecommender:
    """Recommend hairstyles based on face shape and hair type."""
    
    # Recommendation database
    RECOMMENDATIONS = {
        # Oval face - most versatile
        ("Oval", "Straight"): {
            "styles": ["Classic Side Part", "Slick Back", "Pompadour", "Undercut"],
            "tips": "Wajah oval cocok untuk hampir semua gaya. Rambut lurus mudah diatur."
        },
        ("Oval", "Wavy"): {
            "styles": ["Textured Crop", "Messy Fringe", "Natural Waves", "Medium Length"],
            "tips": "Manfaatkan tekstur natural rambut untuk tampilan effortless."
        },
        ("Oval", "Curly"): {
            "styles": ["Curly Fringe", "Medium Curls", "Tapered Curls", "Afro"],
            "tips": "Biarkan curl natural terlihat, jaga kelembaban rambut."
        },
        ("Oval", "Kinky"): {
            "styles": ["High Top", "Twist Out", "Afro", "Fade with Texture"],
            "tips": "Embrace tekstur natural, hidrasi adalah kunci."
        },
        ("Oval", "Dreadlocks"): {
            "styles": ["Free Form Locs", "High Bun Locs", "Loc Fade", "Medium Locs"],
            "tips": "Dreadlocks sangat cocok dengan wajah oval."
        },
        
        # Round face - add height/length
        ("Round", "Straight"): {
            "styles": ["High Fade", "Quiff", "Spiky Top", "Pompadour"],
            "tips": "Tambah volume di atas untuk memperpanjang wajah secara visual."
        },
        ("Round", "Wavy"): {
            "styles": ["Textured Quiff", "Side Part with Volume", "Messy Top"],
            "tips": "Gunakan tekstur wave untuk menambah height."
        },
        ("Round", "Curly"): {
            "styles": ["High Top Curls", "Tapered Sides", "Curly Mohawk", "Fade with Curls"],
            "tips": "Curl di atas membantu memperpanjang wajah."
        },
        ("Round", "Kinky"): {
            "styles": ["High Top Fade", "Flat Top", "Twist Out High", "Temple Fade"],
            "tips": "Fokus pada height untuk balance proporsi wajah."
        },
        ("Round", "Dreadlocks"): {
            "styles": ["High Ponytail Locs", "Top Knot Locs", "Half Up Locs"],
            "tips": "Style locs ke atas untuk menambah panjang visual."
        },
        
        # Square face - soften angles
        ("Square", "Straight"): {
            "styles": ["Textured Crop", "Soft Fringe", "Layered Cut", "Side Swept"],
            "tips": "Gaya dengan tekstur membantu melunakkan sudut wajah."
        },
        ("Square", "Wavy"): {
            "styles": ["Messy Medium Length", "Layered Waves", "Textured Fringe"],
            "tips": "Wave natural membantu soften angular features."
        },
        ("Square", "Curly"): {
            "styles": ["Soft Curls", "Medium Length Curls", "Curly Layers"],
            "tips": "Curls memberi softness pada wajah angular."
        },
        ("Square", "Kinky"): {
            "styles": ["Rounded Afro", "Twist Out", "Soft Texture Top"],
            "tips": "Rounded shapes balance angular jaw."
        },
        ("Square", "Dreadlocks"): {
            "styles": ["Shoulder Length Locs", "Free Form", "Side Swept Locs"],
            "tips": "Locs yang jatuh natural melunakkan sudut."
        },
        
        # Heart face - balance forehead
        ("Heart", "Straight"): {
            "styles": ["Side Swept Bangs", "Chin Length", "Layered Cut", "Fringe"],
            "tips": "Gunakan fringe untuk balance dahi yang lebar."
        },
        ("Heart", "Wavy"): {
            "styles": ["Wavy Fringe", "Chin Length Waves", "Side Part"],
            "tips": "Waves memberi volume di area rahang."
        },
        ("Heart", "Curly"): {
            "styles": ["Curly Bangs", "Jaw Length Curls", "Layered Curls"],
            "tips": "Curls di area rahang balance wajah heart."
        },
        ("Heart", "Kinky"): {
            "styles": ["Defined Twist Out", "Medium Afro", "Side Parted Texture"],
            "tips": "Distribusi volume yang merata membantu balance."
        },
        ("Heart", "Dreadlocks"): {
            "styles": ["Chin Length Locs", "Half Up Style", "Side Locs"],
            "tips": "Locs medium length cocok untuk heart shape."
        },
        
        # Oblong face - add width
        ("Oblong", "Straight"): {
            "styles": ["Full Fringe/Bangs", "Side Volume", "Layered Bob", "Textured Layers"],
            "tips": "Fringe membantu mempersingkat wajah secara visual."
        },
        ("Oblong", "Wavy"): {
            "styles": ["Wavy Bangs", "Side Volume Waves", "Medium Layered"],
            "tips": "Waves di samping menambah width."
        },
        ("Oblong", "Curly"): {
            "styles": ["Curly Bangs", "Wide Curls", "Voluminous Sides"],
            "tips": "Curls memberi width yang dibutuhkan."
        },
        ("Oblong", "Kinky"): {
            "styles": ["Wide Afro", "Bantu Knots", "Voluminous Twist Out"],
            "tips": "Embrace volume untuk balance panjang wajah."
        },
        ("Oblong", "Dreadlocks"): {
            "styles": ["Side Volume Locs", "Crown Wrapped Locs", "Wide Loc Style"],
            "tips": "Style locs untuk menambah width di samping."
        }
    }
    
    # Default recommendation if combination not found
    DEFAULT = {
        "styles": ["Classic Cut", "Textured Crop", "Natural Style"],
        "tips": "Konsultasikan dengan hairstylist untuk rekomendasi personal."
    }
    
    def get_recommendation(self, face_shape, hair_type):
        """Get hairstyle recommendation based on face shape and hair type.
        
        Args:
            face_shape: Face shape string (Oval, Round, Square, Heart, Oblong)
            hair_type: Hair type string (Straight, Wavy, Curly, Kinky, Dreadlocks)
            
        Returns:
            dict: Contains 'styles' list and 'tips' string
        """
        key = (face_shape, hair_type)
        return self.RECOMMENDATIONS.get(key, self.DEFAULT)
    
    def get_all_for_face_shape(self, face_shape):
        """Get all recommendations for a face shape (all hair types).
        
        Args:
            face_shape: Face shape string
            
        Returns:
            dict: Hair type -> recommendation mapping
        """
        results = {}
        for (fs, ht), rec in self.RECOMMENDATIONS.items():
            if fs == face_shape:
                results[ht] = rec
        return results
    
    def get_all_for_hair_type(self, hair_type):
        """Get all recommendations for a hair type (all face shapes).
        
        Args:
            hair_type: Hair type string
            
        Returns:
            dict: Face shape -> recommendation mapping
        """
        results = {}
        for (fs, ht), rec in self.RECOMMENDATIONS.items():
            if ht == hair_type:
                results[fs] = rec
        return results
