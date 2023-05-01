from pydantic import BaseModel

class DataModel(BaseModel):

# Estas varibles permiten que la librería pydantic haga el parseo entre el Json recibido y el modelo declarado.
    id:int
    title: str
    rating: int
    review_text: str
    location: str
    hotel: str
#Esta función retorna los nombres de las columnas correspondientes con el modelo esxportado en joblib.
    def columns(self):
        return ["","title","rating", "review_text","location","hotel"]
