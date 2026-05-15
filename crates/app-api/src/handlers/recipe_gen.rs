use actix_web::{post, web, HttpResponse};
use std::sync::Arc;

use cookest_shared::errors::AppError;
use crate::middleware::auth::AuthenticatedUser;
use crate::services::recipe_gen::{GenerateRecipeRequest, RecipeGenService};

/// POST /api/recipes/generate
///
/// Generate a new recipe using AI.  The generate→score→refine loop runs
/// silently on the server (up to 3 iterations) and returns the best result.
#[post("/api/recipes/generate")]
pub async fn generate_recipe(
    user: AuthenticatedUser,
    service: web::Data<Arc<RecipeGenService>>,
    body: web::Json<GenerateRecipeRequest>,
) -> Result<HttpResponse, AppError> {
    let result = service.generate(user.id, body.into_inner()).await?;
    Ok(HttpResponse::Ok().json(result))
}

pub fn configure_recipe_gen(cfg: &mut web::ServiceConfig) {
    cfg.service(generate_recipe);
}
