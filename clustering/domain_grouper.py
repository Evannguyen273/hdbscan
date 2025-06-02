# clustering/domain_grouper.py
# Updated for new config structure and cumulative training approach
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from collections import Counter, defaultdict
import re

from config.config import get_config

class DomainGrouper:
    """
    Domain grouper for organizing incidents by technical domains.
    Helps with domain-aware clustering and model versioning.
    """
    
    def __init__(self, config=None):
        """Initialize domain grouper with updated config system"""
        self.config = config if config is not None else get_config()
        
        # Domain grouping statistics
        self.grouping_stats = {
            "incidents_grouped": 0,
            "domains_identified": 0,
            "unclassified_incidents": 0,
            "grouping_operations": 0
        }
        
        # Get domain configuration
        self.domain_config = self._get_domain_config()
        
        # Initialize domain patterns and keywords
        self._init_domain_patterns()
        
        logging.info("Domain grouper initialized with %d domain categories", 
                    len(self.domain_config["domain_keywords"]))
    
    def _get_domain_config(self) -> Dict[str, Any]:
        """Get domain grouping configuration with defaults"""
        clustering_config = self.config.clustering.config
        
        return {
            "enable_domain_grouping": clustering_config.get('enable_domain_grouping', True),
            "min_incidents_per_domain": clustering_config.get('min_incidents_per_domain', 10),
            "domain_keywords": clustering_config.get('domain_keywords', self._get_default_domain_keywords()),
            "use_description_field": clustering_config.get('use_description_field', True),
            "use_summary_field": clustering_config.get('use_summary_field', True),
            "case_sensitive": clustering_config.get('case_sensitive', False),
            "require_multiple_keywords": clustering_config.get('require_multiple_keywords', False),
            "default_domain": clustering_config.get('default_domain', 'General')
        }
    
    def _get_default_domain_keywords(self) -> Dict[str, List[str]]:
        """Get default domain keywords for technical categorization"""
        return {
            "Network": [
                "network", "connection", "connectivity", "routing", "switch", "router", 
                "firewall", "vpn", "dns", "dhcp", "ip", "ethernet", "wifi", "wireless",
                "bandwidth", "latency", "packet", "tcp", "udp", "ping", "telnet", "ssh"
            ],
            "Database": [
                "database", "sql", "query", "table", "index", "oracle", "mysql", "postgresql",
                "mongodb", "redis", "connection pool", "transaction", "deadlock", "backup",
                "replication", "schema", "stored procedure", "trigger", "view"
            ],
            "Application": [
                "application", "app", "software", "service", "api", "web service", "microservice",
                "java", "python", "nodejs", "dotnet", ".net", "php", "ruby", "go", "scala",
                "spring", "hibernate", "rest", "soap", "json", "xml", "mvc", "framework"
            ],
            "Infrastructure": [
                "server", "hardware", "cpu", "memory", "disk", "storage", "virtualization",
                "vmware", "hyper-v", "docker", "kubernetes", "container", "cloud", "aws",
                "azure", "gcp", "load balancer", "proxy", "cache", "cdn", "monitoring"
            ],
            "Security": [
                "security", "authentication", "authorization", "ssl", "tls", "certificate",
                "encryption", "decryption", "vulnerability", "patch", "antivirus", "malware",
                "firewall rules", "access control", "identity", "ldap", "active directory",
                "oauth", "saml", "penetration", "intrusion"
            ],
            "Operating_System": [
                "windows", "linux", "unix", "solaris", "aix", "rhel", "ubuntu", "centos",
                "debian", "kernel", "process", "thread", "file system", "permissions",
                "cron", "systemd", "service", "daemon", "registry", "environment variable"
            ],
            "Middleware": [
                "middleware", "message queue", "mq", "jms", "rabbitmq", "kafka", "activemq",
                "web server", "apache", "nginx", "iis", "tomcat", "websphere", "weblogic",
                "jboss", "wildfly", "application server", "esb", "integration"
            ],
            "Data_Processing": [
                "etl", "data pipeline", "batch", "streaming", "hadoop", "spark", "hive",
                "pig", "hdfs", "yarn", "mapreduce", "elasticsearch", "solr", "lucene",
                "data warehouse", "olap", "reporting", "analytics", "machine learning"
            ]
        }
    
    def _init_domain_patterns(self):
        """Initialize regex patterns for domain matching"""
        self.domain_patterns = {}
        
        for domain, keywords in self.domain_config["domain_keywords"].items():
            # Create regex pattern for each domain
            escaped_keywords = [re.escape(keyword) for keyword in keywords]
            pattern_string = r'\b(?:' + '|'.join(escaped_keywords) + r')\b'
            
            if not self.domain_config["case_sensitive"]:
                self.domain_patterns[domain] = re.compile(pattern_string, re.IGNORECASE)
            else:
                self.domain_patterns[domain] = re.compile(pattern_string)
    
    def group_incidents_by_domain(self, incidents: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Group incidents by technical domain.
        
        Args:
            incidents: DataFrame with incident data
            
        Returns:
            Dictionary mapping domain names to incident DataFrames
        """
        grouping_start = datetime.now()
        
        logging.info("Grouping %d incidents by domain", len(incidents))
        
        if not self.domain_config["enable_domain_grouping"]:
            logging.info("Domain grouping disabled, returning all incidents as 'General'")
            return {"General": incidents}
        
        try:
            # Add domain classification to incidents
            incidents_with_domains = self._classify_incident_domains(incidents)
            
            # Group by domain
            domain_groups = {}
            domain_counts = incidents_with_domains['domain'].value_counts()
            
            for domain in domain_counts.index:
                domain_incidents = incidents_with_domains[
                    incidents_with_domains['domain'] == domain
                ].copy()
                
                # Check minimum incident requirement
                min_incidents = self.domain_config["min_incidents_per_domain"]
                if len(domain_incidents) >= min_incidents:
                    # Remove the temporary domain column
                    if 'domain' in domain_incidents.columns:
                        domain_incidents = domain_incidents.drop('domain', axis=1)
                    domain_groups[domain] = domain_incidents
                    
                    logging.info("Domain '%s': %d incidents", domain, len(domain_incidents))
                else:
                    logging.warning("Domain '%s' has insufficient incidents (%d < %d), merging with General",
                                  domain, len(domain_incidents), min_incidents)
                    
                    # Merge with General domain
                    if 'General' not in domain_groups:
                        domain_groups['General'] = pd.DataFrame()
                    
                    domain_incidents_clean = domain_incidents.drop('domain', axis=1, errors='ignore')
                    domain_groups['General'] = pd.concat([
                        domain_groups['General'], 
                        domain_incidents_clean
                    ], ignore_index=True)
            
            # Update statistics
            grouping_duration = datetime.now() - grouping_start
            self.grouping_stats["incidents_grouped"] += len(incidents)
            self.grouping_stats["domains_identified"] += len(domain_groups)
            self.grouping_stats["grouping_operations"] += 1
            
            logging.info("Domain grouping completed: %d domains identified in %.2f seconds",
                        len(domain_groups), grouping_duration.total_seconds())
            
            return domain_groups
            
        except Exception as e:
            logging.error("Domain grouping failed: %s", str(e))
            # Fallback: return all incidents as General
            return {"General": incidents}
    
    def _classify_incident_domains(self, incidents: pd.DataFrame) -> pd.DataFrame:
        """Classify incidents into technical domains"""
        incidents_copy = incidents.copy()
        domains = []
        
        for _, incident in incidents_copy.iterrows():
            # Combine text fields for domain classification
            text_for_classification = self._extract_text_for_classification(incident)
            
            # Classify domain
            domain = self._classify_single_incident(text_for_classification)
            domains.append(domain)
        
        incidents_copy['domain'] = domains
        
        # Log domain distribution
        domain_counts = pd.Series(domains).value_counts()
        logging.info("Domain distribution: %s", domain_counts.to_dict())
        
        return incidents_copy
    
    def _extract_text_for_classification(self, incident: pd.Series) -> str:
        """Extract text fields for domain classification"""
        text_parts = []
        
        # Use description field if available and enabled
        if self.domain_config["use_description_field"]:
            description_fields = ['description', 'Description', 'incident_description', 'summary']
            for field in description_fields:
                if field in incident and pd.notna(incident[field]):
                    text_parts.append(str(incident[field]))
                    break
        
        # Use summary field if available and enabled
        if self.domain_config["use_summary_field"]:
            summary_fields = ['summary', 'Summary', 'processed_text', 'text_summary']
            for field in summary_fields:
                if field in incident and pd.notna(incident[field]):
                    text_parts.append(str(incident[field]))
                    break
        
        # Also check other text fields
        text_fields = ['title', 'subject', 'problem_description', 'resolution', 'notes']
        for field in text_fields:
            if field in incident and pd.notna(incident[field]):
                text_parts.append(str(incident[field]))
        
        return ' '.join(text_parts)
    
    def _classify_single_incident(self, text: str) -> str:
        """Classify a single incident into a domain"""
        if not text or not text.strip():
            return self.domain_config["default_domain"]
        
        domain_scores = {}
        
        # Score each domain based on keyword matches
        for domain, pattern in self.domain_patterns.items():
            matches = pattern.findall(text)
            
            if self.domain_config["require_multiple_keywords"]:
                # Require at least 2 different keywords for classification
                unique_matches = set(match.lower() for match in matches)
                score = len(unique_matches) if len(unique_matches) >= 2 else 0
            else:
                # Count all keyword occurrences
                score = len(matches)
            
            if score > 0:
                domain_scores[domain] = score
        
        # Return domain with highest score
        if domain_scores:
            best_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
            return best_domain
        else:
            return self.domain_config["default_domain"]
    
    def get_domain_statistics(self, incidents: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed statistics about domain distribution"""
        if not self.domain_config["enable_domain_grouping"]:
            return {"domain_grouping_enabled": False}
        
        try:
            incidents_with_domains = self._classify_incident_domains(incidents)
            domain_counts = incidents_with_domains['domain'].value_counts()
            
            stats = {
                "domain_grouping_enabled": True,
                "total_incidents": len(incidents),
                "domains_found": len(domain_counts),
                "domain_distribution": domain_counts.to_dict(),
                "largest_domain": domain_counts.index[0] if len(domain_counts) > 0 else None,
                "smallest_domain": domain_counts.index[-1] if len(domain_counts) > 0 else None
            }
            
            # Calculate domain coverage metrics
            min_incidents = self.domain_config["min_incidents_per_domain"]
            valid_domains = domain_counts[domain_counts >= min_incidents]
            
            stats.update({
                "domains_above_threshold": len(valid_domains),
                "incidents_in_valid_domains": valid_domains.sum(),
                "domain_coverage_percentage": (valid_domains.sum() / len(incidents)) * 100
            })
            
            return stats
            
        except Exception as e:
            logging.error("Failed to calculate domain statistics: %s", str(e))
            return {"error": str(e)}
    
    def predict_incident_domain(self, incident_text: str) -> Tuple[str, Dict[str, int]]:
        """
        Predict domain for a single incident.
        
        Args:
            incident_text: Text content of the incident
            
        Returns:
            Tuple of (predicted_domain, domain_scores)
        """
        if not self.domain_config["enable_domain_grouping"]:
            return self.domain_config["default_domain"], {}
        
        domain_scores = {}
        
        for domain, pattern in self.domain_patterns.items():
            matches = pattern.findall(incident_text)
            
            if self.domain_config["require_multiple_keywords"]:
                unique_matches = set(match.lower() for match in matches)
                score = len(unique_matches) if len(unique_matches) >= 2 else 0
            else:
                score = len(matches)
            
            domain_scores[domain] = score
        
        # Find best domain
        if any(score > 0 for score in domain_scores.values()):
            best_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
        else:
            best_domain = self.domain_config["default_domain"]
        
        return best_domain, domain_scores
    
    def add_custom_domain_keywords(self, domain: str, keywords: List[str]) -> bool:
        """
        Add custom keywords for a domain.
        
        Args:
            domain: Domain name
            keywords: List of keywords to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if domain not in self.domain_config["domain_keywords"]:
                self.domain_config["domain_keywords"][domain] = []
            
            # Add new keywords (avoid duplicates)
            existing_keywords = set(
                kw.lower() for kw in self.domain_config["domain_keywords"][domain]
            )
            
            new_keywords = [
                kw for kw in keywords 
                if kw.lower() not in existing_keywords
            ]
            
            self.domain_config["domain_keywords"][domain].extend(new_keywords)
            
            # Reinitialize patterns
            self._init_domain_patterns()
            
            logging.info("Added %d new keywords to domain '%s'", len(new_keywords), domain)
            return True
            
        except Exception as e:
            logging.error("Failed to add custom domain keywords: %s", str(e))
            return False
    
    def get_grouping_statistics(self) -> Dict[str, Any]:
        """Get domain grouping statistics"""
        return self.grouping_stats.copy()
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate domain grouping configuration"""
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Check minimum incidents requirement
        if self.domain_config["min_incidents_per_domain"] <= 0:
            validation_results["errors"].append("min_incidents_per_domain must be positive")
            validation_results["valid"] = False
        
        # Check domain keywords
        if not self.domain_config["domain_keywords"]:
            validation_results["warnings"].append("No domain keywords configured")
        else:
            # Check each domain has keywords
            for domain, keywords in self.domain_config["domain_keywords"].items():
                if not keywords:
                    validation_results["warnings"].append(f"Domain '{domain}' has no keywords")
        
        # Check default domain
        if not self.domain_config["default_domain"]:
            validation_results["errors"].append("default_domain must be specified")
            validation_results["valid"] = False
        
        return validation_results
    
    def reset_statistics(self):
        """Reset domain grouping statistics"""
        self.grouping_stats = {
            "incidents_grouped": 0,
            "domains_identified": 0,
            "unclassified_incidents": 0,
            "grouping_operations": 0
        }
        logging.info("Domain grouper statistics reset")