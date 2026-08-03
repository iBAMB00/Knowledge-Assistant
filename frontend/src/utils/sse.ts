export interface ParsedSseEvent {
  event: string;
  data: string;
}

export async function consumeSse(
  response: Response,
  onEvent: (event: ParsedSseEvent) => void,
): Promise<void> {
  if (!response.body) {
    throw new Error("浏览器未返回流式响应体。");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const result = await reader.read();
    buffer += decoder.decode(result.value, {
      stream: !result.done,
    });

    let boundary = findBoundary(buffer);

    while (boundary) {
      const rawEvent = buffer.slice(0, boundary.index);
      buffer = buffer.slice(
        boundary.index + boundary.length,
      );

      const parsed = parseSseEvent(rawEvent);
      if (parsed) {
        onEvent(parsed);
      }

      boundary = findBoundary(buffer);
    }

    if (result.done) {
      break;
    }
  }

  if (buffer.trim()) {
    const parsed = parseSseEvent(buffer);
    if (parsed) {
      onEvent(parsed);
    }
  }
}

function findBoundary(
  value: string,
): { index: number; length: number } | null {
  const windowsBoundary = value.indexOf("\r\n\r\n");
  const unixBoundary = value.indexOf("\n\n");

  if (windowsBoundary < 0 && unixBoundary < 0) {
    return null;
  }

  if (
    windowsBoundary >= 0 &&
    (unixBoundary < 0 ||
      windowsBoundary < unixBoundary)
  ) {
    return {
      index: windowsBoundary,
      length: 4,
    };
  }

  return {
    index: unixBoundary,
    length: 2,
  };
}

function parseSseEvent(
  rawEvent: string,
): ParsedSseEvent | null {
  let event = "message";
  const dataLines: string[] = [];

  for (const line of rawEvent.split(/\r?\n/)) {
    if (!line || line.startsWith(":")) {
      continue;
    }

    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
      continue;
    }

    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }

  if (dataLines.length === 0) {
    return null;
  }

  return {
    event,
    data: dataLines.join("\n"),
  };
}
