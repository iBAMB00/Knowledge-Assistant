import { http } from "@/api/http";
import type { TokenResponse, UserRecord } from "@/types/knowledge";

export async function registerUser(email: string, password: string): Promise<UserRecord> {
  const response = await http.post<UserRecord>("/auth/register", { email, password });
  return response.data;
}

export async function loginUser(email: string, password: string): Promise<TokenResponse> {
  const response = await http.post<TokenResponse>("/auth/login", { email, password });
  return response.data;
}

export async function getCurrentUser(): Promise<UserRecord> {
  const response = await http.get<UserRecord>("/auth/me");
  return response.data;
}
