"""
Query Manager
Handles query analysis and clarification for vague queries
"""

import re
from typing import Dict, List

class QueryManager:
    """Manages query analysis and clarification"""
    
    def __init__(self):
        # Keywords that indicate specific information needs
        self.specific_keywords = {
            'medication': ['medication', 'medicine', 'drug', 'pill', 'tablet', 'dose', 'dosage', 'antibiotic', 'prescription'],
            'timing': ['when', 'time', 'schedule', 'frequency', 'daily', 'hourly', 'before', 'after', 'next'],
            'diet': ['food', 'diet', 'eat', 'drink', 'avoid', 'restriction', 'meal', 'nutrition'],
            'activity': ['activity', 'exercise', 'rest', 'work', 'physical', 'movement', 'walk', 'lift'],
            'symptoms': ['symptom', 'pain', 'fever', 'bleeding', 'infection', 'complication', 'warning'],
            'follow_up': ['appointment', 'follow-up', 'doctor', 'visit', 'check-up', 'return']
        }
        
        # Vague query patterns
        self.vague_patterns = [
            r'^(what|how|why|tell me|explain|describe)\s+(about|is|are)',
            r'^(what|how)\s+(should|can|do)',
            r'^general',
            r'^everything',
            r'^all',
            r'^summary',
            r'^overview'
        ]
    
    def analyze_query(self, query: str) -> Dict:
        """
        Analyze query to determine if clarification is needed
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with analysis results
        """
        query_lower = query.lower().strip()
        
        # Check if query is too vague
        is_vague = self._is_vague_query(query_lower)
        
        # Check if query is too short
        is_too_short = len(query.split()) < 3
        
        # Check if query lacks specific keywords
        has_specific_keywords = self._has_specific_keywords(query_lower)
        
        needs_clarification = is_vague or (is_too_short and not has_specific_keywords)
        
        if needs_clarification:
            clarification_questions = self._generate_clarification_questions(query_lower)
            return {
                "needs_clarification": True,
                "questions": clarification_questions,
                "reason": self._get_clarification_reason(is_vague, is_too_short, has_specific_keywords)
            }
        
        return {
            "needs_clarification": False,
            "questions": None,
            "reason": None
        }
    
    def _is_vague_query(self, query: str) -> bool:
        """Check if query matches vague patterns"""
        for pattern in self.vague_patterns:
            if re.search(pattern, query):
                return True
        return False
    
    def _has_specific_keywords(self, query: str) -> bool:
        """Check if query contains specific medical keywords"""
        for category, keywords in self.specific_keywords.items():
            if any(keyword in query for keyword in keywords):
                return True
        return False
    
    def _generate_clarification_questions(self, query: str) -> List[str]:
        """
        Generate clarification questions based on query
        
        Args:
            query: User query string
            
        Returns:
            List of clarification questions
        """
        questions = []
        
        # Detect what type of information might be needed
        detected_categories = []
        
        for category, keywords in self.specific_keywords.items():
            if any(keyword in query for keyword in keywords):
                detected_categories.append(category)
        
        # Generate category-specific questions
        if 'medication' in detected_categories or not detected_categories:
            questions.append("Are you asking about medication instructions, dosages, or timing?")
        
        if 'diet' in detected_categories or not detected_categories:
            questions.append("Are you asking about dietary restrictions or food recommendations?")
        
        if 'activity' in detected_categories or not detected_categories:
            questions.append("Are you asking about activity restrictions or exercise guidelines?")
        
        if 'follow_up' in detected_categories or not detected_categories:
            questions.append("Are you asking about follow-up appointments or when to see your doctor?")
        
        # Add general clarification if no specific category detected
        if not detected_categories:
            questions.extend([
                "What specific aspect of your discharge instructions would you like to know about?",
                "For example: medication timing, dietary restrictions, activity guidelines, or follow-up care?"
            ])
        
        # Limit to 3-4 questions
        return questions[:4]
    
    def _get_clarification_reason(self, is_vague: bool, is_too_short: bool, has_keywords: bool) -> str:
        """Get reason for clarification"""
        reasons = []
        
        if is_vague:
            reasons.append("query is too general")
        if is_too_short:
            reasons.append("query is too short")
        if not has_keywords:
            reasons.append("query lacks specific medical keywords")
        
        return ", ".join(reasons) if reasons else "unknown"
    
    def refine_query(self, original_query: str, clarification: str) -> str:
        """
        Refine original query with clarification
        
        Args:
            original_query: Original vague query
            clarification: User's clarification response
            
        Returns:
            Refined query
        """
        # Combine original query with clarification
        refined = f"{original_query} {clarification}".strip()
        return refined


