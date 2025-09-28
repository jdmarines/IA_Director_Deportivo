import pandas as pd

class QuantitativeAgent:
    """
    Este agente se encarga de todo el análisis de datos estructurados (el CSV).
    Carga los datos una vez y proporciona métodos para consultarlos.
    """
    def __init__(self, csv_path):
        """
        El constructor carga el dataset del CSV en un DataFrame de Pandas.
        """
        try:
            print(f"Cargando dataset desde: {csv_path}")
            self.df = pd.read_csv(csv_path)
            print("Dataset cargado exitosamente.")
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo en la ruta: {csv_path}")
            self.df = None

    def get_player_stats(self, player_name: str) -> dict:
        """
        Busca un jugador por su nombre y devuelve todas sus estadísticas.
        """
        if self.df is None:
            return {"error": "El dataset no está cargado."}
        
        player_data = self.df[self.df['Player'].str.contains(player_name, case=False, na=False)]
        
        if player_data.empty:
            return {"error": f"No se encontraron datos para el jugador: {player_name}"}
        
        return player_data.iloc[0].to_dict()

    def find_top_players(self, metric: str, top_n: int = 5) -> list:
        """
        Encuentra los 'top_n' jugadores en una métrica específica (ej: 'Gls', 'Ast', 'xG').
        """
        if self.df is None:
            return [{"error": "El dataset no está cargado."}]
            
        if metric not in self.df.columns:
            return [{"error": f"La métrica '{metric}' no existe en el dataset."}]

        top_players_df = self.df.sort_values(by=metric, ascending=False).head(top_n)
        
        return top_players_df[['Player', 'Squad', 'Age', metric]].to_dict('records')
