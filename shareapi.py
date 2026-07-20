# from basiclibray import *
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi import APIRouter

router = APIRouter()
# app = FastAPI()

templates = Jinja2Templates(directory="templates")


# Example
# Replace with your PostgreSQL query
# def get_product(product_id):

#     conn = only_connection()

#     cur = conn.cursor()

#     cur.execute("""
#         SELECT
#             title,
#             retail_price,
#             image
#         FROM products
#         WHERE id=%s
#     """,(product_id,))

#     row = cur.fetchone()

#     return {

#         "id": product_id,

#         "title": row[0],

#         "price": f"{row[1]} RWF",

#         "image": row[2]

#     }


def get_product(product_id: int):

    return {
        "id": product_id,
        "title": "iPhone 15 Pro",
        "price": "850,000 RWF",
        "image": "https://res.cloudinary.com/demo/image/upload/iphone.jpg"
    }



@router.get("/shareapi/{product_id}", response_class=HTMLResponse)
async def share_product(request: Request, product_id: int):

    product = get_product(product_id)

    return templates.TemplateResponse(
        "share.html",
        {
            "request": request,
            "product": product
        }
    )
