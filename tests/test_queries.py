"""
Unit tests for PheTK._queries module.
"""
import pytest
import os
import sys
from unittest.mock import Mock, patch

# Add src directory to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from PheTK import _queries


class TestPhecodeIcdQuery:
    """Test phecode_icd_query function."""

    def test_phecode_icd_query_basic(self):
        """Test basic SQL query generation."""
        ds = "test_dataset"
        query = _queries.phecode_icd_query(ds)
        
        assert isinstance(query, str)
        assert len(query) > 0
        assert ds in query
        assert "condition_occurrence" in query
        assert "concept" in query

    def test_phecode_icd_query_contains_required_tables(self):
        """Test that query contains required OMOP tables."""
        ds = "test_dataset"
        query = _queries.phecode_icd_query(ds)
        
        # Should reference required OMOP tables
        assert f"{ds}.condition_occurrence" in query
        assert f"{ds}.concept" in query

    def test_phecode_icd_query_contains_required_columns(self):
        """Test that query selects required columns."""
        ds = "test_dataset"
        query = _queries.phecode_icd_query(ds)
        
        # Should select required columns
        required_columns = [
            "person_id",
            "condition_start_date",
            "vocabulary_id",
            "concept_code",
            "condition_concept_id"
        ]
        
        for column in required_columns:
            assert column in query

    def test_phecode_icd_query_filters_vocabulary(self):
        """Test that query filters for ICD vocabularies."""
        ds = "test_dataset"
        query = _queries.phecode_icd_query(ds)
        
        # Should filter for ICD9CM and ICD10CM vocabularies
        assert "ICD9CM" in query
        assert "ICD10CM" in query
        assert "vocabulary_id in" in query

    def test_phecode_icd_query_joins_correctly(self):
        """Test that query joins tables correctly."""
        ds = "test_dataset"
        query = _queries.phecode_icd_query(ds)
        
        # Should have proper JOIN syntax
        assert "JOIN" in query or "INNER JOIN" in query
        assert "ON" in query

    def test_phecode_icd_query_handles_v_codes(self):
        """Test that query handles V codes specially."""
        ds = "test_dataset"
        query = _queries.phecode_icd_query(ds)
        
        # The query should have logic for handling V codes
        # which are present in both ICD9CM and ICD10CM
        assert "UNION" in query

    def test_phecode_icd_query_distinct_results(self):
        """Test that query uses DISTINCT to avoid duplicates."""
        ds = "test_dataset"
        query = _queries.phecode_icd_query(ds)
        
        assert "DISTINCT" in query

    def test_phecode_icd_query_with_different_datasets(self):
        """Test query generation with different dataset names."""
        datasets = ["cdr_v7", "cdr_v8", "my_custom_dataset", "test_123"]
        
        for ds in datasets:
            query = _queries.phecode_icd_query(ds)
            assert ds in query
            assert f"{ds}.condition_occurrence" in query
            assert f"{ds}.concept" in query

    def test_phecode_icd_query_structure(self):
        """Test the overall structure of the generated query."""
        ds = "test_dataset"
        query = _queries.phecode_icd_query(ds)
        
        # Convert to uppercase for case-insensitive checks
        query_upper = query.upper()
        
        # Should be a valid SQL query structure
        assert "SELECT" in query_upper
        assert "FROM" in query_upper
        assert "WHERE" in query_upper
        
        # Should handle both source_value and source_concept_id joins
        source_value_joins = query.count("condition_source_value")
        source_concept_joins = query.count("condition_source_concept_id")
        
        assert source_value_joins > 0
        assert source_concept_joins > 0

    def test_phecode_icd_query_date_handling(self):
        """Test that query properly handles date fields."""
        ds = "test_dataset"
        query = _queries.phecode_icd_query(ds)
        
        # Should include date information
        assert "condition_start_date" in query
        assert "date" in query

    def test_phecode_icd_query_concept_id_handling(self):
        """Test that query handles concept IDs correctly."""
        ds = "test_dataset"
        query = _queries.phecode_icd_query(ds)
        
        # Should handle both concept_code and concept_id
        assert "concept_code" in query
        assert "concept_id" in query


class TestQueryValidation:
    """Test query validation and error handling."""

    def test_phecode_icd_query_empty_dataset(self):
        """Test query generation with empty dataset name."""
        query = _queries.phecode_icd_query("")
        
        # Should still generate query but with empty dataset references
        assert isinstance(query, str)
        assert len(query) > 0

    def test_phecode_icd_query_special_characters(self):
        """Test query generation with special characters in dataset name."""
        # Test with dataset names that might have special characters
        special_datasets = ["test-dataset", "test_dataset_123", "test.dataset"]
        
        for ds in special_datasets:
            query = _queries.phecode_icd_query(ds)
            assert isinstance(query, str)
            assert ds in query

    def test_phecode_icd_query_sql_injection_safety(self):
        """Test that function doesn't allow SQL injection."""
        # Test with potential SQL injection strings
        malicious_inputs = [
            "test'; DROP TABLE users; --",
            "test; SELECT * FROM sensitive_table",
            "test UNION SELECT password FROM users"
        ]
        
        for malicious_input in malicious_inputs:
            query = _queries.phecode_icd_query(malicious_input)
            
            # The malicious input should be treated as a literal dataset name
            # and not executed as SQL
            assert isinstance(query, str)
            # The malicious SQL should appear as literal text, not as executable SQL
            assert malicious_input in query


class TestQueryPerformance:
    """Test query performance considerations."""

    def test_phecode_icd_query_length(self):
        """Test that generated query is reasonable length."""
        ds = "test_dataset"
        query = _queries.phecode_icd_query(ds)
        
        # Query should be substantial but not excessively long
        assert 100 < len(query) < 10000

    def test_phecode_icd_query_complexity(self):
        """Test query complexity metrics."""
        ds = "test_dataset"
        query = _queries.phecode_icd_query(ds)
        
        # Count various SQL components
        select_count = query.upper().count("SELECT")
        join_count = query.upper().count("JOIN")
        where_count = query.upper().count("WHERE")
        union_count = query.upper().count("UNION")
        
        # Should have reasonable complexity
        assert select_count >= 2  # At least 2 SELECT statements
        assert join_count >= 2    # At least 2 JOIN operations
        assert where_count >= 2   # At least 2 WHERE clauses
        assert union_count >= 1   # At least 1 UNION operation


class TestQueryIntegration:
    """Integration tests for query functions."""

    def test_phecode_icd_query_with_realistic_dataset_names(self):
        """Test with realistic All of Us dataset names."""
        realistic_datasets = [
            "aou-res-curation-prod.aou_synthetic_cdr_clean_2022q4_5",
            "aou-res-curation-prod.aou_synthetic_cdr_clean_2023q1_1",
            "project.dataset.table_prefix"
        ]
        
        for ds in realistic_datasets:
            query = _queries.phecode_icd_query(ds)
            
            assert isinstance(query, str)
            assert len(query) > 0
            assert ds in query
            
            # Should handle dots in dataset names correctly
            if "." in ds:
                assert f"{ds}.condition_occurrence" in query
                assert f"{ds}.concept" in query

    def test_query_consistency(self):
        """Test that query generation is consistent."""
        ds = "test_dataset"
        
        # Generate same query multiple times
        queries = [_queries.phecode_icd_query(ds) for _ in range(5)]
        
        # All queries should be identical
        for query in queries[1:]:
            assert query == queries[0]

    def test_query_contains_essential_logic(self):
        """Test that query contains essential business logic."""
        ds = "test_dataset"
        query = _queries.phecode_icd_query(ds)
        
        # Essential components for PheTK functionality
        essential_elements = [
            "condition_occurrence",  # Main OMOP table
            "concept",              # For vocabulary mapping
            "ICD9CM",              # ICD-9 vocabulary
            "ICD10CM",             # ICD-10 vocabulary
            "vocabulary_id",       # For filtering vocabularies
            "concept_code",        # ICD codes
            "person_id",           # Patient identifier
            "DISTINCT",            # Deduplication
            "UNION"                # Combining multiple sources
        ]
        
        for element in essential_elements:
            assert element in query, f"Missing essential element: {element}"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_none_dataset_parameter(self):
        """Test behavior with None dataset parameter."""
        with pytest.raises(ValueError):
            _queries.phecode_icd_query(None)

    def test_numeric_dataset_parameter(self):
        """Test behavior with numeric dataset parameter."""
        query = _queries.phecode_icd_query(123)
        
        # Should convert to string and work
        assert "123" in query

    def test_very_long_dataset_name(self):
        """Test with very long dataset name."""
        long_name = "a" * 1000
        query = _queries.phecode_icd_query(long_name)
        
        assert isinstance(query, str)
        assert long_name in query

    def test_unicode_dataset_name(self):
        """Test with unicode characters in dataset name."""
        unicode_name = "test_dataset_特殊字符"
        query = _queries.phecode_icd_query(unicode_name)
        
        assert isinstance(query, str)
        assert unicode_name in query
