import { clerkMiddleware } from "@clerk/nextjs/server";

// Next.js 16 renamed "Middleware" to "Proxy" (file convention only --
// functionality is unchanged). This just needs to run on every route so
// Clerk can attach auth context; the app itself stays fully public
// (SignedOut users can still use it, scoped to standard access -- see
// api.py's ANONYMOUS_USER).
export default clerkMiddleware();

export const config = {
  matcher: ["/((?!_next|.*\\..*).*)"],
};
