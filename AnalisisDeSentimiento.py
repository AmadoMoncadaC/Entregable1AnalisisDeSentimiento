# Importacion de bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Lista de stopwords en español (puedes ampliarla si es necesario)
stopwords_espanol = [
    'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'se', 'del', 'las', 'un', 'por', 'para', 'con', 'una',
    'su', 'no', 'al', 'lo', 'como', 'más', 'pero', 'sus', 'le', 'ya', 'o', 'este', 'si', 'me', 'es', 'porque',
    'esta', 'entre', 'cuando', 'muy', 'sin', 'sobre', 'ser', 'también', 'otros', 'fue', 'ha', 'está', 'yo',
    'hasta', 'hay', 'donde', 'quien', 'después', 'te', 'ni', 'nos', 'durante', 'todos', 'algunos', 'este', 'él',
    'ellas', 'ante', 'ese', 'esto', 'esa', 'esos', 'esas', 'unos', 'una', 'su', 'de', 'todo', 'mismo', 'ya'
]

# Datos (reseñas de productos)
data = {
    'review': [
        'Este producto es increíble, lo recomiendo mucho',
        'Muy mala calidad, no lo volvería a comprar',
        'Excelente relación calidad-precio, muy satisfecho',
        'No me gustó el producto, llegó defectuoso',
        'Muy bueno, cumple con lo prometido',
        'Horrible, no funciona como debería',
        'Perfecto para lo que necesitaba, lo volveré a comprar',
        'No vale lo que cuesta, decepcionado',
        'Pesimo producto malo',
        'No funciona nada bien, muy malo',
        # Generación de datos sintéticos (positivos)
        'Me encantó este producto, es fantástico',
        'Excelente calidad, superó mis expectativas',
        'Lo compraría nuevamente, es perfecto para lo que busco',
        'Es el mejor producto que he comprado, me sorprendió mucho',
        'Estoy muy feliz con la compra, totalmente recomendado',
        'Muy útil, y el precio es justo para lo que ofrece',
        # Generación de datos sintéticos (negativos)
        'El producto llegó roto y no funcionaba, pésima calidad',
        'Muy decepcionado, no lo recomiendo para nada',
        'El artículo no vale lo que cuesta, no es bueno',
        'No cumple con lo prometido, muy insatisfecho',
        'No lo volvería a comprar, fue una mala experiencia',
        'Producto defectuoso, muy mala compra',
        'No funciona como se describe, no lo recomiendo',
        'Compre este producto y fue una total decepción',
        # Nuevas reseñas
        'Este producto superó mis expectativas, excelente calidad y muy útil',  # Positiva
        'Totalmente decepcionado, llegó roto y no sirve',  # Negativa
        'Muy buen desempeño, lo recomiendo totalmente',  # Positiva
        'No lo compraría de nuevo, el material es muy frágil',  # Negativa
        'Increíble relación calidad-precio, me encanta',  # Positiva
        'Producto defectuoso, no funciona como se espera',  # Negativa
        'El producto es exactamente lo que necesitaba, fantástico',  # Positiva
        'Una gran decepción, no cumple lo prometido',  # Negativa
        'El servicio de entrega fue rápido y el producto es excelente',  # Positiva
        'No vale la pena, se rompió al poco tiempo de uso',  # Negativa
        'Me sorprendió lo bien que funciona, totalmente recomendado',  # Positiva
        'Malísimo, no tiene buena calidad, muy decepcionante',  # Negativa
        'Es el mejor producto que he comprado, sin dudas',  # Positiva
        'Muy mala experiencia, llegó tarde y mal embalado',  # Negativa
        'Me encanta, superó mis expectativas en todo',  # Positiva
        'Definitivamente no lo volveré a comprar, no funciona',  # Negativa
        'Muy bueno, cumple con todo lo prometido',  # Positiva
        'Producto de mala calidad, no lo recomiendo',  # Negativa
        'Es un producto increíble, lo usaré todos los días',  # Positiva
        'No es lo que esperaba, no lo recomiendo en absoluto',  # Negativa
        'Es perfecto para mi casa, me encanta',  # Positiva
        'Decepcionante, no funciona como se describe',  # Negativa
        'Excelente, vale cada centavo que pagué',  # Positiva
        'No lo recomiendo para nada, muy malo',  # Negativa
        'Me sorprendió lo bien que funciona, perfecto para lo que buscaba',  # Positiva
        'No sirve para nada, muy deficiente',  # Negativa
        'Muy buena compra, quedé satisfecho con el producto',  # Positiva
        'Malo, no lo compres',
        'Muy buen producto',
        'No me gusto',
        'Muy mal producto',
        #
        "Este producto es increíble. Superó mis expectativas.",
        "La calidad es pésima. Se rompió al primer uso.",
        "Me encanta. Funciona a la perfección y es muy fácil de usar.",
        "No lo recomiendo para nada. Es una estafa.",
        "Es justo lo que necesitaba. Cumple su función a la perfección.",
        "Después de usarlo un tiempo, puedo decir que es bastante bueno.",
        "Es el peor producto que he comprado. No pierdan su dinero.",
        "Muy contento con la compra. El envío fue rápido y el producto es excelente.",
        "No me impresionó. Es bastante mediocre para el precio que tiene.",
        "Simplemente perfecto. Lo recomiendo al 100%.",
        # Positivas
        "¡Absolutamente fantástico! Superó todas mis expectativas.",
        "Excelente producto. Funciona de maravilla y es muy fácil de usar.",
        "Muy contento con esta compra. La calidad es excepcional.",
        "Lo recomiendo al 100%. Un producto de alta calidad.",
        "Me encanta este producto. Es justo lo que estaba buscando.",
        "Impresionado con la calidad y el rendimiento. ¡Vale la pena cada centavo!",
        "Un producto increíblemente útil y bien diseñado.",
        "Funciona a la perfección. No tengo ninguna queja.",
        "La mejor compra que he hecho en mucho tiempo. ¡Muy satisfecho!",
        "Este producto es una joya. Lo recomiendo sin dudarlo.",
        "Satisfecho al 100 con la compra. El producto es tal como se describe.",
        "¡Increíble! Superó mis expectativas. Lo recomiendo ampliamente.",
        "Es justo lo que necesitaba. Cumple su función a la perfección.",
        "Después de usarlo un tiempo, puedo decir que es bastante bueno.",
        "Muy contento con la compra. El envío fue rápido y el producto es excelente.",
        "Simplemente perfecto. Lo recomiendo al 100%.",
        "Un producto maravilloso. Me ha facilitado mucho la vida.",
        "Calidad superior. No me arrepiento de la compra.",
        "¡Excelente relación calidad-precio! Muy contento con el producto.",
        "Funciona de maravilla y es muy intuitivo. ¡Lo recomiendo!",
        "Este producto es maravilloso. Lo recomiendo ampliamente.",
        "Me ha encantado este producto. Es justo lo que buscaba.",
        "¡Increíble calidad! Superó mis expectativas por completo.",
        "Muy fácil de usar y con resultados excelentes.",
        "Estoy realmente impresionado con este producto. ¡Altamente recomendado!",
        "Un producto excepcional que cumple con todas sus promesas.",
        "Sin duda, una de mis mejores compras. ¡Estoy muy contento!",
        "Este producto es simplemente genial. ¡Lo recomiendo a todos!",
        "Me ha sorprendido gratamente. ¡Una excelente adquisición!",
        "Estoy fascinado con este producto. ¡Es perfecto en todos los sentidos!",
        # Negativas
        "Una decepción total. No funciona como se esperaba.",
        "Pésima calidad. Se rompió al poco tiempo de usarlo.",
        "No lo recomiendo para nada. Es una pérdida de dinero.",
        "Muy insatisfecho con esta compra. El producto llegó defectuoso.",
        "Este producto es una estafa. No pierdan su tiempo ni su dinero.",
        "No cumple con las expectativas. Es bastante mediocre.",
        "La peor compra que he hecho en mucho tiempo. ¡Decepcionante!",
        "El producto llegó en mal estado. No lo recomiendo.",
        "No me impresionó para nada. Es de muy baja calidad.",
        "Una gran decepción. No vale la pena el precio que tiene.",
        "Mala calidad y mal funcionamiento. ¡No lo compren!",
        "Este producto es un desastre. No funciona correctamente.",
        "Completamente insatisfecho. El producto no sirve para nada.",
        "Una experiencia terrible. El producto es de pésima calidad.",
        "No malgasten su dinero en esto. Es un producto deficiente.",
        "Muy decepcionado. El producto no cumple con lo prometido.",
        "Pésimo producto. No lo recomiendo bajo ninguna circunstancia.",
        "Un verdadero fiasco. No funciona como debería.",
        "Horrible producto. No pierdan su tiempo ni su dinero.",
        "Completamente arrepentido de haber comprado este producto.",
        "Este producto es una basura. No sirve para nada.",
        "Muy mala calidad. Se dañó a los pocos días de uso.",
        "No lo compren. Es una completa estafa.",
        "Una total decepción. No funciona como se anuncia.",
        "Pésimo servicio y peor producto. No lo recomiendo en absoluto.",
        "Este producto es un fraude. No cumple con ninguna expectativa.",
        "Una pérdida de tiempo y dinero. No lo recomiendo para nada.",
        "Completamente insatisfecho. El producto es defectuoso.",
        "Una experiencia horrible. El producto es de muy mala calidad.",
        "No malgasten su dinero en esto. Es un producto inservible.",
        #
        "¡Absolutamente fantástico! Superó todas mis expectativas. Lo recomiendo sin dudarlo.",  # 1
        "Decepcionante. No funciona como se describe. Una pérdida de dinero.",  # 0
        "Muy buen producto, fácil de usar y con excelentes resultados. Estoy muy satisfecho.",  # 1
        "Pésima calidad. Se rompió a los pocos días de uso. No lo compren.",  # 0
        "Me encanta este producto. Es justo lo que necesitaba. Funciona a la perfección.",  # 1
        "Una estafa total. No pierdan su tiempo ni su dinero. No sirve para nada.",  # 0
        "Es un producto excelente. Cumple con su función y es muy duradero. Lo recomiendo.",  # 1
        "Mala compra. No lo recomiendo para nada. No vale la pena el precio.",  # 0
        "Estoy muy contento con esta compra. El producto es de alta calidad y funciona de maravilla.",  # 1
        "Terrible. No funciona y el servicio al cliente es pésimo. No lo recomiendo.",  # 0
        "Increíble. Me ha facilitado mucho la vida. Lo recomiendo al 100%.",  # 1
        "Muy malo. No sirve para lo que se anuncia. Me siento estafado.",  # 0
        "Excelente producto. Lo recomiendo ampliamente. Superó mis expectativas.",  # 1
        "Decepcionado. Esperaba mucho más. No cumple con lo prometido.",  # 0
        "Funciona de maravilla. Estoy muy contento con la compra. Lo recomiendo sin duda.",  # 1
        "No lo compren. Es una porquería. Se rompió al instante.",  # 0
        "Es justo lo que buscaba. Cumple su función a la perfección. Muy satisfecho.",  # 1
        "Una completa basura. No sirve para nada. No pierdan su dinero.",  # 0
        "Muy buen producto. Lo recomiendo ampliamente. Es muy útil y práctico.",  # 1
        "Mala calidad. No dura nada. Se rompió a la primera semana. No lo recomiendo.",  # 0
        "¡Me encanta! Es perfecto. Funciona de maravilla y es muy fácil de usar.",  # 1
        "Una estafa. No funciona como debería. No lo compren bajo ninguna circunstancia.",  # 0
        "Es un producto fantástico. Lo recomiendo a todo el mundo. Es muy útil y práctico.",  # 1
        "Muy decepcionado. No vale la pena el precio. No cumple con las expectativas.",  # 0
        "Funciona muy bien. Estoy muy contento con la compra. Lo recomiendo sin dudarlo.",  # 1
        "No lo recomiendo para nada. Es una pérdida de tiempo y dinero. No funciona.",  # 0
        "Es un producto excelente. Lo recomiendo ampliamente. Es muy eficiente y práctico.",  # 1
        "Mala inversión. No lo compren. No sirve para lo que se anuncia. Me siento engañado.",  # 0
        "Estoy muy satisfecho con esta compra. El producto es de muy buena calidad y funciona a la perfección.",  # 1
        "Pésimo producto. No lo compren. Se rompió al instante. No vale la pena.",  # 0
        "La entrega fue rapidísima y el producto llegó en perfectas condiciones. Muy contento.",  # 1
        "El manual de instrucciones es confuso y el montaje fue complicado. No me ha gustado la experiencia.",  # 0
        "Por el precio que tiene, esperaba mucha más calidad. Me ha decepcionado un poco.",  # 0
        "Es muy intuitivo y fácil de usar, incluso para personas que no son muy tecnológicas.",  # 1
        "El servicio de atención al cliente fue excelente, me resolvieron todas mis dudas rápidamente.",  # 1
        "El producto llegó con un defecto de fábrica. Estoy gestionando la devolución.",  # 0
        "Me ha sorprendido gratamente la durabilidad de este producto. Lo recomiendo sin duda.",  # 1
        "No cumple con las expectativas que tenía. Me esperaba algo mucho mejor.",  # 0
        "Es un producto muy versátil y se adapta a mis necesidades a la perfección.",  # 1
        "El diseño es muy elegante y moderno. Queda perfecto en cualquier ambiente.",  # 1
        "Después de usarlo durante un tiempo, puedo decir que es una buena inversión.",  # 1
        "No me ha gustado nada la calidad de los materiales. Se ven muy frágiles.",  # 0
        "Es muy práctico y fácil de transportar. Ideal para llevar de viaje.",  # 1
        "El producto es tal y como se describe en la página web. Muy satisfecho con la compra.",  # 1
        "He tenido problemas para configurarlo. Las instrucciones no son claras.",  # 0
        "Es un producto innovador y con muchas funcionalidades interesantes.",  # 1
        "No me ha convencido. Hay otras opciones mejores en el mercado.",  # 0
        "Es perfecto para el uso que le doy. Lo recomiendo totalmente.",  # 1
        "El embalaje llegó dañado y el producto tenía algunos rasguños.",  # 0
        "Me ha solucionado un problema que tenía desde hace tiempo. Muy útil.",  # 1
        #
        "Me ha encantado este producto. Es justo lo que necesitaba para mi trabajo.",  # 1
        "La relación calidad-precio es inmejorable. Muy recomendable.",  # 1
        "Es muy intuitivo y fácil de usar, incluso para personas mayores.",  # 1
        "El diseño es elegante y moderno. Queda perfecto en mi salón.",  # 1
        "La entrega fue rapidísima y el embalaje impecable. Muy satisfecho.",  # 1
        "Es un producto muy completo y con muchas funcionalidades útiles.",  # 1
        "Funciona a la perfección y es muy silencioso. Lo recomiendo sin duda.",  # 1
        "La batería dura muchísimo. Es un gran punto a favor.",  # 1
        "Es muy ligero y cómodo de llevar. Ideal para viajes.",  # 1
        "Me ha solucionado un problema que tenía desde hace tiempo. Muy útil.",  # 1
        "Es resistente y duradero. Ideal para uso diario.",  # 1
        "No me ha gustado nada. Es de muy mala calidad y no funciona bien.",  # 0
        "Me arrepiento de haber comprado este producto. No lo recomiendo para nada.",  # 0
        "Es muy complicado de configurar y el manual es confuso.",  # 0
        "El producto llegó defectuoso y estoy teniendo problemas con la devolución.",  # 0
        "La calidad de los materiales es pésima. Se ve muy frágil.",  # 0
        "Es demasiado caro para lo que ofrece. No vale la pena.",  # 0
        "No cumple con las expectativas que tenía. Me esperaba mucho más.",  # 0
        "La batería dura muy poco y tarda mucho en cargar.",  # 0
        "Es muy ruidoso y molesto. No lo puedo usar en espacios tranquilos.",  # 0
        "El servicio de atención al cliente es pésimo. No me han ayudado en nada.",  # 0
        "Después de un mes de uso, se ha roto sin motivo aparente.",  # 0
        "Es un producto muy básico que no ofrece nada nuevo.",  # 0
        "El diseño es anticuado y poco atractivo.",  # 0
        "No es compatible con mi dispositivo. Una gran decepción.",  # 0
        "Es demasiado grande y pesado. No es práctico para transportar.",  # 0
        "El olor que desprende es muy desagradable.",  # 0
        "He tenido problemas de conexión desde el primer día.",  # 0
        "Es muy difícil de limpiar y mantener.",  # 0
        "El sonido es de muy mala calidad. No lo recomiendo para escuchar música.",  # 0
        "La imagen se ve borrosa y con poca nitidez.",  # 0
        "No es compatible con los accesorios que ya tenía.",  # 0
        "Es un producto muy frágil que se rompe con facilidad.",  # 0
        "No me ha gustado la textura de los materiales.",  # 0
        "Es un producto muy contaminante y poco respetuoso con el medio ambiente.",  # 0
        "Es inseguro y poco confiable. No lo recomiendo para niños.",  # 0
        "Es aburrido y poco entretenido.",  # 0
        # 0 (Si fuera comida/bebida)
        "No me ha gustado el sabor. Es demasiado artificial.",
        "Es muy pegajoso y difícil de quitar.",  # 0
        "Mancha la ropa y otras superficies.",  # 0
        "Es un producto innovador que me ha sorprendido gratamente.",  # 1
        "La relación calidad-precio es excelente. Lo recomiendo sin dudarlo.",  # 1
        "Es muy fácil de usar y las instrucciones son claras y concisas.",  # 1
        "El embalaje es muy cuidado y el producto llegó en perfectas condiciones.",  # 1
        "Es un producto muy versátil que se adapta a diferentes necesidades.",  # 1
        "Es muy potente y eficiente. Cumple su función a la perfección.",  # 1
        "Es muy compacto y ocupa poco espacio. Ideal para espacios reducidos.",  # 1
        "Es muy silencioso y no molesta para nada.",  # 1
        "Es un producto muy duradero que me durará mucho tiempo.",  # 1
        "Es muy cómodo y ergonómico. Se adapta perfectamente a mi cuerpo.",  # 1
        #
        "Este producto es una maravilla. Me ha simplificado la vida enormemente.",  # 1
        "Estoy muy decepcionado. No funciona como esperaba y el material es de baja calidad.",  # 0
        "Excelente compra. Superó mis expectativas en todos los sentidos.",  # 1
        "No lo recomiendo para nada. Es una pérdida de dinero y tiempo.",  # 0
        "Me encanta lo fácil que es de usar y lo bien que funciona. ¡Muy recomendable!",  # 1
        "Es una estafa. No compren esto, no sirve para nada.",  # 0
        "La calidad es impresionante. Se nota que está hecho con buenos materiales.",  # 1
        "Es bastante mediocre para el precio que tiene. No vale la pena.",  # 0
        "Estoy fascinado con este producto. Es justo lo que necesitaba.",  # 1
        "Es terrible. Se rompió a la primera semana de uso. No lo compren.",  # 0
        "Me ha sorprendido gratamente. Funciona de maravilla y es muy práctico.",  # 1
        "Es una porquería. No funciona correctamente y es muy frágil.",  # 0
        "Es un producto innovador y con muchas funciones útiles. Lo recomiendo.",  # 1
        "Me arrepiento de haberlo comprado. No cumple con lo que promete.",  # 0
        "Es perfecto para mis necesidades. Funciona a la perfección y es muy fácil de usar.",  # 1
        "Es una completa decepción. No lo recomiendo bajo ninguna circunstancia.",  # 0
        "Es un producto fantástico. Lo recomiendo a todo el mundo. Es muy útil y práctico.",  # 1
        "Muy decepcionado. No vale la pena el precio. No cumple con las expectativas.",  # 0
        "Funciona muy bien y es muy fácil de instalar. Estoy muy contento con la compra.",  # 1
        "No lo compren. Es una pérdida de tiempo y dinero. No funciona como debería.",  # 0
        "Es un producto excelente y muy eficiente. Lo recomiendo ampliamente.",  # 1
        "Mala inversión. No lo compren. No sirve para lo que se anuncia. Me siento engañado.",  # 0
        "Estoy muy satisfecho con esta compra. El producto es de muy buena calidad y funciona a la perfección.",  # 1
        "Pésimo producto. No lo compren. Se rompió al instante. No vale la pena.",  # 0
        "¡Increíble! Me ha solucionado muchos problemas. Lo recomiendo al 100%.",  # 1
        "Muy malo. No sirve para nada. Es una estafa total. No lo recomiendo a nadie.",  # 0
        "Excelente calidad y muy fácil de usar. Lo recomiendo sin dudarlo.",  # 1
        "Decepcionante. No funciona como esperaba. No lo recomiendo para nada.",  # 0
        "Funciona a la perfección y es muy silencioso. Estoy muy contento con la compra.",  # 1
        "No lo compren. Es una porquería y no sirve para nada. Es una estafa.",  # 0
        "Es justo lo que buscaba. Cumple con su función a la perfección y es muy práctico.",  # 1
        "Una completa basura. No lo recomiendo para nada. Es una pérdida de dinero total.",  # 0
        "Muy buen producto. Es muy útil y práctico. Lo recomiendo ampliamente a todos.",  # 1
        "Mala calidad. No dura nada. Se rompió a la primera semana de uso. No lo recomiendo en absoluto.",  # 0
        "¡Me encanta! Es perfecto. Funciona de maravilla y es muy fácil de usar. Lo recomiendo a todos mis amigos.",  # 1
        "Una estafa. No funciona como se anuncia. No lo compren bajo ninguna circunstancia. Es un fraude.",  # 0
        "Es un producto fantástico y muy innovador. Lo recomiendo a todo el mundo. Es muy útil y práctico para el día a día.",  # 1
        "Muy decepcionado. No vale la pena el precio. No cumple con las expectativas que tenía puestas en él.",  # 0
        "Funciona muy bien y es muy intuitivo. Estoy muy contento con la compra. Lo recomiendo sin dudarlo a nadie.",  # 1
        "No lo recomiendo para nada. Es una pérdida de tiempo y dinero. No funciona correctamente. Es un timo.",  # 0
        "Es un producto excelente y muy eficiente. Lo recomiendo ampliamente a todos los que necesiten algo así.",  # 1
        "Mala inversión. No lo compren. No sirve para lo que se anuncia. Me siento engañado y estafado.",  # 0
        "Estoy muy satisfecho con esta compra. El producto es de muy buena calidad y funciona a la perfección. Lo recomiendo.",  # 1
        "Pésimo producto. No lo compren. Se rompió al instante. No vale la pena para nada. Es una basura.",  # 0
        "¡Increíble! Me ha solucionado muchos problemas que tenía. Lo recomiendo al 100 a todo el mundo.",  # 1
        "Muy malo. No sirve para nada. Es una estafa total. No lo recomiendo ni a mi peor enemigo.",  # 0
        "Excelente calidad. Lo recomiendo sin dudarlo. Es justo lo que necesitaba para mi trabajo y para mi vida personal.",  # 1
        "Decepcionante. No funciona como esperaba. No lo recomiendo para nada. Me siento muy frustrado.",  # 0
        "Funciona a la perfección y es muy silencioso. Estoy muy contento con la compra y lo recomiendo a todo el mundo.",  # 1
        "Este producto es justo lo que necesitaba, me ha facilitado mucho el trabajo. Lo recomiendo al 100%",  # 1
        #
        "Es un producto robusto y confiable. Me ha impresionado su durabilidad.",  # 1
        "La peor compra que he hecho en mucho tiempo. No sirve para nada.",  # 0
        "Excelente relación calidad-precio. Lo recomiendo sin dudarlo.",  # 1
        "Una auténtica decepción. No funciona como se anuncia y es muy frágil.",  # 0
        "Me encanta lo práctico y funcional que es. ¡Una gran adquisición!",  # 1
        "Un completo desastre. No lo compren, es una estafa.",  # 0
        "La calidad de los materiales es excepcional. Se nota la diferencia.",  # 1
        "Es bastante mediocre para el precio que tiene. No lo recomiendo.",  # 0
        "Estoy muy satisfecho con este producto. Cumple con todas mis expectativas.",  # 1
        "Es pésimo. Se dañó a los pocos días de uso. No vale la pena.",  # 0
        "Me ha sorprendido gratamente su rendimiento. ¡Muy recomendable!",  # 1
        "Es una porquería. No funciona correctamente y es muy inestable.",  # 0
        "Un producto innovador con características únicas. ¡Lo recomiendo!",  # 1
        "Me arrepiento de haberlo comprado. No cumple con su función principal.",  # 0
        "Es perfecto para mi estilo de vida. ¡Funciona de maravilla!",  # 1
        "Una total pérdida de dinero. No lo compren bajo ninguna circunstancia.",  # 0
        "Es un producto fantástico que supera todas las expectativas. ¡Muy útil!",  # 1
        "Muy decepcionado. El precio es excesivo para lo que ofrece.",  # 0
        "Funciona a la perfección y es muy intuitivo. ¡Estoy encantado!",  # 1
        "No pierdan su tiempo ni su dinero. Este producto es un timo.",  # 0
        "Un producto excelente y muy versátil. ¡Lo recomiendo a todo el mundo!",  # 1
        "Mala inversión. No lo compren, no vale lo que cuesta.",  # 0
        "Estoy muy contento con esta compra. La calidad es insuperable.",  # 1
        "Pésimo producto. Se rompió al instante. ¡No lo compren, por favor!",  # 0
        "¡Increíble! Me ha resuelto muchos problemas. ¡Lo recomiendo al 100% a todos!",  # 1
        "Muy malo. No sirve para nada. ¡Es una estafa! No lo compren jamás.",  # 0
        "Excelente calidad y diseño innovador. ¡Lo recomiendo sin reservas!",  # 1
        "Decepcionante. No funciona como esperaba. ¡No lo recomiendo en absoluto!",  # 0
        "Funciona a la perfección y es muy silencioso. ¡Estoy muy feliz con mi compra!",  # 1
        "No lo compren. Es una verdadera porquería. ¡Es un fraude total!",  # 0
        "Es justo lo que necesitaba. Cumple su función a la perfección. ¡Muy satisfecho!",  # 1
        "Una completa estafa. No lo recomiendo para nada. ¡Es tirar el dinero!",  # 0
        "Muy buen producto. Es muy práctico y fácil de usar. ¡Lo recomiendo a todos!",  # 1
        "Mala calidad. No dura nada. Se averió a la primera semana. ¡No lo compren bajo ningún concepto!",  # 0
        "¡Me encanta! Es perfecto. Funciona de maravilla y es muy intuitivo. ¡Lo recomiendo a mis amigos y familiares!",  # 1
        "Una estafa. No funciona como se anuncia. ¡No lo compren! Es un engaño.",  # 0
        "Es un producto fantástico y muy práctico para el día a día. ¡Lo recomiendo encarecidamente!",  # 1
        "Muy decepcionado. No cumple con mis expectativas. ¡No lo recomiendo en absoluto!",  # 0
        "Funciona muy bien y es muy fácil de configurar. ¡Estoy muy contento con la compra y lo recomiendo!",  # 1
        "No lo recomiendo para nada. Es una pérdida de tiempo y dinero. ¡No funciona para nada!",  # 0
        "Es un producto excelente y muy eficiente para mi trabajo. ¡Lo recomiendo ampliamente a profesionales!",  # 1
        "Mala inversión. No lo compren. No sirve para lo que se anuncia. ¡Me siento estafado y engañado!",  # 0
        "Estoy muy satisfecho con esta compra. El producto es de alta calidad y funciona de maravilla. ¡Lo recomiendo a todo el mundo!",  # 1
        "Pésimo producto. No lo compren. Se rompió al instante. ¡No vale la pena en absoluto! ¡Es una porquería!",  # 0
        "¡Increíble! Me ha solucionado todos mis problemas. ¡Lo recomiendo al 100 a quien necesite algo así!",  # 1
        "Muy malo. No sirve para nada. ¡Es una estafa total! No lo recomiendo ni a mi peor enemigo. ¡Es un timo!",  # 0
        "Excelente calidad y muy fácil de usar. ¡Lo recomiendo sin dudarlo a cualquier persona!",  # 1
        "Decepcionante. No funciona como esperaba y además llegó con un defecto. ¡No lo recomiendo en absoluto!",  # 0
        "Funciona a la perfección y es muy silencioso. ¡Estoy muy contento con la compra y lo recomiendo a todo el mundo sin excepción!",  # 1
        "Este producto es justo lo que necesitaba, me ha facilitado mucho la vida y el trabajo. ¡Lo recomiendo al 100 a todos mis conocidos!",  # 1
        #
        "La tela de esta camisa es increíblemente suave. ¡Muy cómoda!",  # 1
        "Los pantalones me quedaron demasiado ajustados. Tuve que devolverlos.",  # 0
        "El vestido es precioso, ideal para una fiesta. ¡Me encantó cómo me quedó!",  # 1
        "La calidad de la costura es pésima. Se descosió al primer lavado.",  # 0
        "La chaqueta es perfecta para el invierno. ¡Muy abrigadora y con estilo!",  # 1
        "El color del jersey no es igual al de la foto. Me decepcionó un poco.",  # 0
        "La falda tiene un corte muy favorecedor. ¡Me siento muy cómoda con ella!",  # 1
        "La talla del abrigo es mucho más grande de lo que esperaba. Tuve que pedir una talla menos.",  # 0
        "Los zapatos son muy cómodos y elegantes. ¡Ideales para el trabajo!",  # 1
        "Las medias se rompieron al primer uso. ¡Muy mala calidad!",  # 0
        "La blusa es muy versátil. Se puede usar tanto para ocasiones formales como informales.",  # 1
        "El cinturón es demasiado rígido. No es cómodo de usar.",  # 0
        "El traje me quedó como un guante. ¡Perfecto para una boda!",  # 1
        "La gorra es demasiado pequeña. No me queda bien.",  # 0
        "Los guantes son muy cálidos y suaves. ¡Ideales para el frío!",  # 1
        "La bufanda es demasiado delgada. No abriga mucho.",  # 0
        "El chaleco es muy ligero y cómodo. ¡Perfecto para entretiempo!",  # 1
        "Los calcetines son demasiado cortos. Se me bajan constantemente.",  # 0
        "La ropa interior es muy suave y cómoda. ¡La recomiendo!",  # 1
        "El pijama es demasiado caluroso. No es cómodo para dormir.",  # 0
        "Me encanta el diseño de esta sudadera. Es muy original.",  # 1
        "Los shorts son demasiado cortos. No me siento cómoda usándolos.",  # 0
        "La camiseta tiene un estampado muy bonito. ¡Me gusta mucho!",  # 1
        "El bañador se transparenta cuando está mojado. ¡Qué vergüenza!",  # 0
        "Los leggins son muy cómodos y se adaptan muy bien al cuerpo.",  # 1
        "Los pantalones vaqueros me quedan perfectos. ¡Son muy favorecedores!",  # 1
        "El top es demasiado escotado. No es de mi estilo.",  # 0
        "El kimono es precioso y muy elegante. ¡Ideal para el verano!",  # 1
        "El vestido de noche es espectacular. ¡Me sentí como una princesa!",  # 1
        "La ropa deportiva es muy cómoda y transpirable. ¡Perfecta para el gimnasio!",  # 1
        "El jersey me pica mucho. No puedo usarlo directamente sobre la piel.",  # 0
        "La camisa se arruga con mucha facilidad. Es un fastidio tener que plancharla constantemente.",  # 0
        "Los zapatos son muy bonitos, pero me lastimaron los pies la primera vez que los usé.",  # 0
        "El abrigo es muy pesado. No es práctico para el día a día.",  # 0
        "La falda es demasiado larga. Tuve que mandarla a cortar.",  # 0
        "Me encanta el color de esta blusa. Es muy favorecedor para mi tono de piel.",  # 1
        "El cinturón se rompió a los pocos días de uso. ¡Muy mala calidad!",  # 0
        "El traje me queda un poco grande. Tendré que ajustarlo.",  # 0
        "La gorra es de muy buena calidad y me protege muy bien del sol.",  # 1
        "Los guantes son muy elegantes y combinan con cualquier outfit.",  # 1
        "La bufanda tiene un diseño muy original. ¡Me encanta!",  # 1
        "El chaleco es muy práctico para llevar encima de otras prendas.",  # 1
        "Los calcetines son muy suaves y cómodos. ¡Los recomiendo!",  # 1
        "La ropa interior es de buena calidad y se ajusta muy bien.",  # 1
        "El pijama es muy fresco y cómodo para el verano.",  # 1
        "Me encanta el estilo de esta sudadera. Es muy moderna.",  # 1
        "Los shorts son muy cómodos para ir a la playa.",  # 1
        "La camiseta tiene un tacto muy agradable.",  # 1
        "El bañador se seca muy rápido. ¡Ideal para las vacaciones!",  # 1
        "Los leggins son perfectos para hacer ejercicio.",  # 1
        "Los pantalones vaqueros son muy resistentes y duraderos.",  # 1
        "El top es muy bonito y elegante. ¡Ideal para una ocasión especial!",  # 1
        "El kimono es muy versátil y se puede usar de muchas maneras diferentes.",  # 1
        "El vestido de noche es muy cómodo de llevar a pesar de ser elegante.",  # 1
        "La ropa deportiva me da mucha libertad de movimiento.",  # 1
        # Chatgpt
        'Buena relación calidad-precio, muy contento',
        'Recomiendo este artículo, es fantástico',
        'No funcionó como se esperaba, decepcionante',
        'El servicio al cliente fue terrible',
        'Muy mala calidad, no lo recomendaría',
        'Una pérdida de dinero, muy insatisfecho',
        'Buena relación calidad-precio, muy contento',
        'El envío llegó tarde y el producto estaba dañado',
        'Buena relación calidad-precio, muy contento',
        'Es perfecto, justo lo que necesitaba',
        'Estoy muy satisfecho con la compra',
        'Recomiendo este artículo, es fantástico',
        'Una pérdida de dinero, muy insatisfecho',
        'No funcionó como se esperaba, decepcionante',
        'Muy mala calidad, no lo recomendaría',
        'Recomiendo este artículo, es fantástico',
        'Recomiendo este artículo, es fantástico',
        'No funcionó como se esperaba, decepcionante',
        'El servicio al cliente fue terrible',
        'No funcionó como se esperaba, decepcionante',
        'Recomiendo este artículo, es fantástico',
        'El envío llegó tarde y el producto estaba dañado',
        'El envío llegó tarde y el producto estaba dañado',
        'El envío llegó tarde y el producto estaba dañado',
        'Estoy muy satisfecho con la compra',
        'Estoy muy satisfecho con la compra',
        'Recomiendo este artículo, es fantástico',
        'Es perfecto, justo lo que necesitaba',
        'Estoy muy satisfecho con la compra',
        'Muy mala calidad, no lo recomendaría',
        'Es perfecto, justo lo que necesitaba',
        'El servicio al cliente fue terrible',
        'El servicio al cliente fue terrible',
        'El producto es excelente, superó mis expectativas',
        'Estoy muy satisfecho con la compra',
        'El servicio al cliente fue terrible',
        'Estoy muy satisfecho con la compra',
        'El servicio al cliente fue terrible',
        'Muy mala calidad, no lo recomendaría',
        'Estoy muy satisfecho con la compra',
        'Muy mala calidad, no lo recomendaría',
        'Una pérdida de dinero, muy insatisfecho',
        'Muy mala calidad, no lo recomendaría',
        'Buena relación calidad-precio, muy contento',
        'El producto es excelente, superó mis expectativas',
        'Recomiendo este artículo, es fantástico',
        'Estoy muy satisfecho con la compra',
        'El servicio al cliente fue terrible',
        'Es perfecto, justo lo que necesitaba',
        'Estoy muy satisfecho con la compra',
        'El servicio al cliente fue terrible',
        'Estoy muy satisfecho con la compra',
        'No funcionó como se esperaba, decepcionante',
        'El servicio al cliente fue terrible',
        'Una pérdida de dinero, muy insatisfecho',
        'El producto es excelente, superó mis expectativas',
        'Buena relación calidad-precio, muy contento',
        'El envío llegó tarde y el producto estaba dañado',
        'Una pérdida de dinero, muy insatisfecho',
        'Una pérdida de dinero, muy insatisfecho',
        'El servicio al cliente fue terrible',
        'Muy mala calidad, no lo recomendaría',
        'Recomiendo este artículo, es fantástico',
        'El producto es excelente, superó mis expectativas',
        'Recomiendo este artículo, es fantástico',
        'No funcionó como se esperaba, decepcionante',
        'No funcionó como se esperaba, decepcionante',
        'Una pérdida de dinero, muy insatisfecho',
        'El servicio al cliente fue terrible',
        'El servicio al cliente fue terrible',
        'Estoy muy satisfecho con la compra',
        'El envío llegó tarde y el producto estaba dañado',
        'Una pérdida de dinero, muy insatisfecho',
        'No funcionó como se esperaba, decepcionante',
        'Buena relación calidad-precio, muy contento',
        'Estoy muy satisfecho con la compra',
        'Recomiendo este artículo, es fantástico',
        'Estoy muy satisfecho con la compra',
        'El producto es excelente, superó mis expectativas',
        'Estoy muy satisfecho con la compra',
        'Estoy muy satisfecho con la compra',
        'Es perfecto, justo lo que necesitaba',
        'El producto es excelente, superó mis expectativas',
        'El envío llegó tarde y el producto estaba dañado',
        'Buena relación calidad-precio, muy contento',
        'Estoy muy satisfecho con la compra',
        'Una pérdida de dinero, muy insatisfecho',
        'El servicio al cliente fue terrible',
        'Estoy muy satisfecho con la compra',
        'El servicio al cliente fue terrible',
        'No funcionó como se esperaba, decepcionante',
        'Una pérdida de dinero, muy insatisfecho',
        'Estoy muy satisfecho con la compra',
        'No funcionó como se esperaba, decepcionante',
        'Una pérdida de dinero, muy insatisfecho',
        'El servicio al cliente fue terrible',
        'Una pérdida de dinero, muy insatisfecho',
        'Una pérdida de dinero, muy insatisfecho',
        'Estoy muy satisfecho con la compra',
        'Recomiendo este artículo, es fantástico',
        #

    ],
    'sentiment': [
        1, 0, 1, 0, 1, 0, 1, 0, 0, 0,  # 1 = positiva, 0 = negativa
        1, 1, 1, 1, 1, 1,  # Positivas adicionales
        0, 0, 0, 0, 0, 0, 0, 0,  # Negativas adicionales
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        1, 0, 0,
        1, 0, 1, 0, 1, 1, 0, 1, 0, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1,

    ]
}

# Convertir el diccionario en un DataFrame
df = pd.DataFrame(data)

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.3, random_state=42)

# Agregar algunas palabras clave negativas específicas al vectorizador
stopwords_adicionales = ['pesimo', 'malo',
                         'horrible', 'defectuoso', 'decepcionado']

# Crear una lista de stopwords que incluye las palabras en español y las adicionales
stop_words_completo = stopwords_espanol + \
    stopwords_adicionales  # Concatenar las stopwords

# Aplicar TF-IDF vectorización para convertir el texto en vectores
# Añadir las palabras clave negativas y las stopwords en español
tfidf = TfidfVectorizer(stop_words=stop_words_completo, lowercase=True)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Entrenar un modelo de regresión logística
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Hacer predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test_tfidf)

# Evaluar la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Mostrar un reporte detallado de clasificación
print(classification_report(y_test, y_pred))

# Función para predecir el sentimiento de una nueva reseña


def predecir_sentimiento(reseña):
    # Transformar la nueva reseña utilizando el vectorizador TF-IDF
    reseña_tfidf = tfidf.transform([reseña])

    # Realizar la predicción con el modelo entrenado
    prediccion = model.predict(reseña_tfidf)

    # Interpretar el resultado
    if prediccion[0] == 1:
        return "Positiva"
    else:
        return "Negativa"


# Pedir al usuario que ingrese una reseña
reseña_usuario = input(
    "Ingresa tu reseña de producto para analizar el sentimiento: ")

# Predecir el sentimiento de la reseña ingresada
sentimiento = predecir_sentimiento(reseña_usuario)

# Mostrar el resultado
print(f"La reseña es clasificada como: {sentimiento}")
