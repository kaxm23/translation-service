                return data
            else:
                st.error("Failed to fetch translation statistics")
                return {}
                
        except Exception as e:
            st.error(f"API request failed: {str(e)}")
            return {}

    def _get_available_languages(self, lang_type: str) -> List[str]:
        """Get available languages from history."""
        cache_key = f"languages_{lang_type}"
        
        if cache_key in self.cache:
            cache_time, languages = self.cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_timeout:
                return languages
        
        try:
            response = requests.get(
                f"{self.api_base_url}/translations/languages/",
                headers={"Authorization": f"Bearer {st.session_state.token}"},
                params={"type": lang_type}
            )
            
            if response.status_code == 200:
                languages = response.json()["languages"]
                self.cache[cache_key] = (datetime.now(), languages)
                return languages
            else:
                return []
                
        except Exception as e:
            st.error(f"Failed to fetch languages: {str(e)}")
            return []

    def _export_data(self, data: pd.DataFrame, format: str):
        """Export translation history data."""
        if format == "CSV":
            return data.to_csv(index=False)
        elif format == "Excel":
            return data.to_excel(index=False)
        elif format == "JSON":
            return data.to_json(orient="records", date_format="iso")
        return None

if __name__ == "__main__":
    app = TranslationHistoryApp()
    app.run()