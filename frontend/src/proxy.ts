/**
 * Runs Clerk's middleware on every route so auth context (the signed-in
 * user, if any) is available to the app. Named proxy.ts, not middleware.ts,
 * because Next.js 16 renamed the "Middleware" file convention to "Proxy"
 * (functionality is unchanged). The app itself stays fully public --
 * signed-out users can still use it, scoped to standard access server-side
 * (see api.py's ANONYMOUS_USER).
 */
import { clerkMiddleware } from "@clerk/nextjs/server";

export default clerkMiddleware();

export const config = {
  matcher: ["/((?!_next|.*\\..*).*)"],
};
