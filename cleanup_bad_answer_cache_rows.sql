-- One-off cleanup of answer_cache rows that must not be served, found 2026-10-17 (read-only scan of
-- all 113 rows). Run AFTER answer_cache_versioning.sql. Each step is safe to re-run.
--
-- Under the new server code no pre-existing row can be served at all (they have no prompt_version),
-- so STEP 2 alone is enough to make the cache clean. STEP 1 removes the rows that were actively
-- harmful under the OLD code, so they're gone even if the deploy is delayed or rolled back.
--
--    4  history-dependent key
--    2  AMBIGUOUS clarify reply
--    9  addresses a student by name
--    1  self-correction
--    4  removed 'Chapter: Not available' line
--   36  July (pre-August) format
--   56  total (a row with several problems is listed under its first one)

-- STEP 1a (preview): expect 56.
select count(*) from public.answer_cache where id in (
  -- history-dependent key
  '442617c0-a476-412f-8054-df972e7441b4',  -- 2026-07-08 (also: July (pre-August) format; addresses a student by name) | Hello, Student! 👋 /  / Welcome to NEET-AI! I'm here to help you 
  'baedbd29-da13-43e7-9258-be72a1c854f2',  -- 2026-07-19 (also: July (pre-August) format) | Hey there, Student! 👋 /  / I'm ready to help! 😊 /  / Do you have a *
  'baefec98-9412-4189-a421-66b5163fe9ad',  -- 2026-08-30 | DOUBT_TYPE: conversational /  / Hello! 😊 How can I help you with
  '5902e416-2c82-4cb6-8783-e0524ec0a096',  -- 2026-10-13 | DOUBT_TYPE: conversational /  / Sure — here's the whole conversa
  -- AMBIGUOUS clarify reply
  '487b1190-c4e5-4ad0-bf9b-f60d33552dd7',  -- 2026-08-16 | AMBIGUOUS: yes / CLARIFY_TYPE: topic / QUESTION: What do you mea
  '9ad49418-b0cc-4335-8cc6-36c470fd1551',  -- 2026-08-16 | AMBIGUOUS: yes / CLARIFY_TYPE: topic / QUESTION: What cycle? / OPT
  -- addresses a student by name
  '4e57273e-45f3-472f-b314-39aae120345c',  -- 2026-07-10 (also: July (pre-August) format) | Hey Student! 👋 /  / I see you've sent a greeting, but I didn't g
  '08dbfe17-de83-4a94-a89e-8c5de1e74c4e',  -- 2026-07-19 (also: July (pre-August) format) | Hey Student! 👋 /  / I think there might be a typo there! 😊 /  / Coul
  '4757aefd-a952-4552-904d-d7270751482c',  -- 2026-07-19 (also: July (pre-August) format) | Hey Student! 😊 /  / I see you're just testing me out! That's tot
  '6390aa16-b97c-40f1-ab05-61d46cb85cdc',  -- 2026-07-19 (also: July (pre-August) format) | Hey Student! 😄 /  / I think your keyboard might be having fun! 🎹
  'af3d293c-411e-49c2-9e9f-4221860d008b',  -- 2026-07-21 (also: July (pre-August) format) | Hey Student! 👋 /  / I'm here and ready to help you ace your NEET
  '5f24abc6-1fcb-4f7d-bfd9-a3ab9db48949',  -- 2026-07-25 (also: July (pre-August) format) | Great question, Student! Let me help you with this classic N
  '08339919-9672-4b89-810f-00aed344b3fd',  -- 2026-09-06 (also: removed 'Chapter: Not available' line) | VISUAL_INTENT: no /  / NEET Importance: 3/5 /  / 📚 Chapter: Not avai
  '67e4d023-a085-489d-851d-ec8ed9c165d4',  -- 2026-09-06 | VISUAL_INTENT: no /  / NEET Importance: 5/5 /  / 📚 Chapter: NCERT Cl
  '7c35664e-cd40-4c3d-9c60-41cd9c2b05c6',  -- 2026-09-20 | DOUBT_TYPE: conversational /  / Hey Student! Ready when you are 
  -- self-correction
  '0c1147f7-683b-4652-b93a-fb8b30b7c1d0',  -- 2026-09-06 (also: removed 'Chapter: Not available' line) | NEET Importance: 4/5 /  / 📚 Chapter: Not available — answering f
  -- removed 'Chapter: Not available' line
  '318a3786-0942-4661-a7b3-ce0c4af06ddb',  -- 2026-08-28 | VISUAL_INTENT: no /  / NEET Importance: 4/5 /  / 📚 Chapter: Not avai
  '15f6627b-0b16-4be5-b72b-298289480578',  -- 2026-08-30 | VISUAL_INTENT: no /  / NEET Importance: 5/5 /  / 📚 Chapter: Not avai
  'd5c505ee-2759-4d83-9a6e-545da2f1aac3',  -- 2026-09-07 | VISUAL_INTENT: yes /  / NEET Importance: 3/5 /  / 📚 Chapter: Not ava
  '3c017a8a-e119-440d-9ea2-8a71dbfdbe05',  -- 2026-09-07 | VISUAL_INTENT: no /  / NEET Importance: 5/5 /  / 📚 Chapter: Not avai
  -- July (pre-August) format
  '5b061434-f1be-4346-8d2d-e542431c6a68',  -- 2026-07-08 | NEET Importance: ⭐⭐⭐⭐ (4/5) /  / 📚 Chapter: NCERT Class 9, Chapt
  '0d30aa16-54f0-42a1-94ad-316c7c5fe1ff',  -- 2026-07-08 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: [NCERT Class X, Cha
  '5d577201-14e6-4e05-b73e-476a005013ec',  -- 2026-07-08 | To solve this problem, I need to use the relationship betwee
  'aa96434d-8344-4a0e-a940-5b61044c3760',  -- 2026-07-08 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: NCERT Class XII, Ch
  'b2eb6324-0327-40a2-ba46-9c90c05efccc',  -- 2026-07-10 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: NCERT Class XI, Cha
  '9a938af7-85ab-4d09-9276-fa0ed95712d1',  -- 2026-07-10 | **NEET Importance: ⭐⭐⭐⭐⭐ (5/5)** /  / 📚 **Chapter:** NCERT Class
  'be6a005b-6c56-4ebb-8df3-1dd072ebff59',  -- 2026-07-14 | **NEET Importance:** ⭐⭐⭐⭐⭐ (5/5) /  / 📚 **Chapter:** NCERT Class
  '31cf5fcc-6429-4b66-ab62-0be8db6aad2b',  -- 2026-07-14 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: NCERT Class XI, Cha
  '62ee5eff-ba4d-4d94-ba7d-c209a10eaa6a',  -- 2026-07-14 | **NEET Importance: ⭐⭐⭐⭐⭐ (5/5)** /  / 📚 **Chapter:** NCERT Class
  'aa8bb49c-7d4f-4e42-83c7-eee9f7dceb95',  -- 2026-07-14 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: NCERT Class XI, Cha
  'b1601fac-2efe-423a-b1dd-cdc91a6c094c',  -- 2026-07-19 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: Chemistry - Class 1
  '9cdf5bed-beb2-4a22-941d-e7844f4144d1',  -- 2026-07-19 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: NCERT Class 11, Cha
  '75113171-1d86-4d4c-b813-55c3ca570804',  -- 2026-07-21 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: Physics - Class 11 
  'ea5fff15-2db2-4317-b16b-d1a277ba6424',  -- 2026-07-21 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: NCERT Class XI, Cha
  'd874fa5c-3e2e-4bd4-b35a-ff12940eec7f',  -- 2026-07-21 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: NCERT Class XII, Ch
  '39c53ef1-9258-483d-9aca-2409382ec0c1',  -- 2026-07-21 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: NCERT Class XI, Cha
  '025ad1a2-e5a9-4b5a-83e2-c95a956cce73',  -- 2026-07-21 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: NCERT Class XII, Ch
  '1442309f-aa8b-4bf1-a818-90a59f38c453',  -- 2026-07-21 | **NEET महत्व: ⭐⭐⭐⭐⭐ (5/5)** /  / 📚 **अध्याय:** NCERT कक्षा XI, अ
  '74591c02-fe31-4264-ad67-95ad08968186',  -- 2026-07-21 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: NCERT Class 11, Cha
  '5acd377d-ab45-4ffe-8b7c-22b011b33dad',  -- 2026-07-21 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: NCERT Class XI, Cha
  '26cc60bb-3874-4e99-a736-4995c7e1a9a2',  -- 2026-07-21 | Hello! 👋  /  / I'm **NEET-AI**, your expert tutor for Indian med
  '49424d06-e6ec-4288-9885-9db3e31f0833',  -- 2026-07-22 | NEET Importance: ⭐⭐⭐⭐ (4/5) /  / 📚 Chapter: NCERT Class XI, Chap
  '673f7656-546a-48c3-8863-91cadc0eedbb',  -- 2026-07-25 | Great! Let me reveal the answer and explanation, Student. 🎯 / 
  '7cce4488-061e-45cd-8397-327af3e9d1cb',  -- 2026-07-25 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: NCERT Class XI, Cha
  '19b6042d-eb03-43d1-b9ad-52fe36557801',  -- 2026-07-26 | **NEET Importance: ⭐⭐⭐⭐⭐ (5/5)** /  / 📚 **Chapter:** NCERT Class
  'e98c0255-1501-4205-ae46-ef84e2b303d4',  -- 2026-07-26 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: NCERT Class XI, Cha
  'bd8a408c-6fc3-4799-b723-06504037d6f1',  -- 2026-07-26 | NEET Importance: ⭐⭐⭐⭐ (4/5) /  / 📚 Chapter: [NCERT Class XI, Cha
  '1206a534-13f5-42ba-b36a-cf0540d7d911',  -- 2026-07-26 | NEET Importance: ⭐⭐⭐⭐ (4/5) /  / 📚 Chapter: [NCERT Class XI, Cha
  'cc1571db-cd22-4150-92db-d7742637a7b3',  -- 2026-07-26 | NEET Importance: ⭐⭐⭐⭐ (4/5) /  / 📚 Chapter: NCERT Class 11, Chap
  '86757444-035d-4b4a-8db4-a91cca4f44e7',  -- 2026-07-26 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: NCERT Class XI, Cha
  '60f1ea09-cdb1-475b-9a13-677897bb25ef',  -- 2026-07-26 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: NCERT Class XI, Cha
  '4d653f53-a2e4-4950-a771-876d43b366be',  -- 2026-07-27 | # Fetal Development Flowchart /  / I'd be happy to help, Student
  '52e19aaf-fa7c-4723-91c5-dfffb1775bf9',  -- 2026-07-27 | # Additional Fetal Development Flowcharts /  / Sure, Student! He
  '2647d35a-8ee1-442c-83fc-ae60dce41e6e',  -- 2026-07-27 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: NCERT Class 11 — Bi
  'bdddfb50-513f-4656-842b-8366a2fddcfc',  -- 2026-07-29 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: NCERT Class 11, Cha
  '9fd6c421-6fe4-425d-9fde-d3ac0a35448c'   -- 2026-07-29 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: NCERT Class 11, Cha
);

-- STEP 1b: delete them.
delete from public.answer_cache where id in (
  -- history-dependent key
  '442617c0-a476-412f-8054-df972e7441b4',  -- 2026-07-08 (also: July (pre-August) format; addresses a student by name) | Hello, Student! 👋 /  / Welcome to NEET-AI! I'm here to help you 
  'baedbd29-da13-43e7-9258-be72a1c854f2',  -- 2026-07-19 (also: July (pre-August) format) | Hey there, Student! 👋 /  / I'm ready to help! 😊 /  / Do you have a *
  'baefec98-9412-4189-a421-66b5163fe9ad',  -- 2026-08-30 | DOUBT_TYPE: conversational /  / Hello! 😊 How can I help you with
  '5902e416-2c82-4cb6-8783-e0524ec0a096',  -- 2026-10-13 | DOUBT_TYPE: conversational /  / Sure — here's the whole conversa
  -- AMBIGUOUS clarify reply
  '487b1190-c4e5-4ad0-bf9b-f60d33552dd7',  -- 2026-08-16 | AMBIGUOUS: yes / CLARIFY_TYPE: topic / QUESTION: What do you mea
  '9ad49418-b0cc-4335-8cc6-36c470fd1551',  -- 2026-08-16 | AMBIGUOUS: yes / CLARIFY_TYPE: topic / QUESTION: What cycle? / OPT
  -- addresses a student by name
  '4e57273e-45f3-472f-b314-39aae120345c',  -- 2026-07-10 (also: July (pre-August) format) | Hey Student! 👋 /  / I see you've sent a greeting, but I didn't g
  '08dbfe17-de83-4a94-a89e-8c5de1e74c4e',  -- 2026-07-19 (also: July (pre-August) format) | Hey Student! 👋 /  / I think there might be a typo there! 😊 /  / Coul
  '4757aefd-a952-4552-904d-d7270751482c',  -- 2026-07-19 (also: July (pre-August) format) | Hey Student! 😊 /  / I see you're just testing me out! That's tot
  '6390aa16-b97c-40f1-ab05-61d46cb85cdc',  -- 2026-07-19 (also: July (pre-August) format) | Hey Student! 😄 /  / I think your keyboard might be having fun! 🎹
  'af3d293c-411e-49c2-9e9f-4221860d008b',  -- 2026-07-21 (also: July (pre-August) format) | Hey Student! 👋 /  / I'm here and ready to help you ace your NEET
  '5f24abc6-1fcb-4f7d-bfd9-a3ab9db48949',  -- 2026-07-25 (also: July (pre-August) format) | Great question, Student! Let me help you with this classic N
  '08339919-9672-4b89-810f-00aed344b3fd',  -- 2026-09-06 (also: removed 'Chapter: Not available' line) | VISUAL_INTENT: no /  / NEET Importance: 3/5 /  / 📚 Chapter: Not avai
  '67e4d023-a085-489d-851d-ec8ed9c165d4',  -- 2026-09-06 | VISUAL_INTENT: no /  / NEET Importance: 5/5 /  / 📚 Chapter: NCERT Cl
  '7c35664e-cd40-4c3d-9c60-41cd9c2b05c6',  -- 2026-09-20 | DOUBT_TYPE: conversational /  / Hey Student! Ready when you are 
  -- self-correction
  '0c1147f7-683b-4652-b93a-fb8b30b7c1d0',  -- 2026-09-06 (also: removed 'Chapter: Not available' line) | NEET Importance: 4/5 /  / 📚 Chapter: Not available — answering f
  -- removed 'Chapter: Not available' line
  '318a3786-0942-4661-a7b3-ce0c4af06ddb',  -- 2026-08-28 | VISUAL_INTENT: no /  / NEET Importance: 4/5 /  / 📚 Chapter: Not avai
  '15f6627b-0b16-4be5-b72b-298289480578',  -- 2026-08-30 | VISUAL_INTENT: no /  / NEET Importance: 5/5 /  / 📚 Chapter: Not avai
  'd5c505ee-2759-4d83-9a6e-545da2f1aac3',  -- 2026-09-07 | VISUAL_INTENT: yes /  / NEET Importance: 3/5 /  / 📚 Chapter: Not ava
  '3c017a8a-e119-440d-9ea2-8a71dbfdbe05',  -- 2026-09-07 | VISUAL_INTENT: no /  / NEET Importance: 5/5 /  / 📚 Chapter: Not avai
  -- July (pre-August) format
  '5b061434-f1be-4346-8d2d-e542431c6a68',  -- 2026-07-08 | NEET Importance: ⭐⭐⭐⭐ (4/5) /  / 📚 Chapter: NCERT Class 9, Chapt
  '0d30aa16-54f0-42a1-94ad-316c7c5fe1ff',  -- 2026-07-08 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: [NCERT Class X, Cha
  '5d577201-14e6-4e05-b73e-476a005013ec',  -- 2026-07-08 | To solve this problem, I need to use the relationship betwee
  'aa96434d-8344-4a0e-a940-5b61044c3760',  -- 2026-07-08 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: NCERT Class XII, Ch
  'b2eb6324-0327-40a2-ba46-9c90c05efccc',  -- 2026-07-10 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: NCERT Class XI, Cha
  '9a938af7-85ab-4d09-9276-fa0ed95712d1',  -- 2026-07-10 | **NEET Importance: ⭐⭐⭐⭐⭐ (5/5)** /  / 📚 **Chapter:** NCERT Class
  'be6a005b-6c56-4ebb-8df3-1dd072ebff59',  -- 2026-07-14 | **NEET Importance:** ⭐⭐⭐⭐⭐ (5/5) /  / 📚 **Chapter:** NCERT Class
  '31cf5fcc-6429-4b66-ab62-0be8db6aad2b',  -- 2026-07-14 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: NCERT Class XI, Cha
  '62ee5eff-ba4d-4d94-ba7d-c209a10eaa6a',  -- 2026-07-14 | **NEET Importance: ⭐⭐⭐⭐⭐ (5/5)** /  / 📚 **Chapter:** NCERT Class
  'aa8bb49c-7d4f-4e42-83c7-eee9f7dceb95',  -- 2026-07-14 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: NCERT Class XI, Cha
  'b1601fac-2efe-423a-b1dd-cdc91a6c094c',  -- 2026-07-19 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: Chemistry - Class 1
  '9cdf5bed-beb2-4a22-941d-e7844f4144d1',  -- 2026-07-19 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: NCERT Class 11, Cha
  '75113171-1d86-4d4c-b813-55c3ca570804',  -- 2026-07-21 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: Physics - Class 11 
  'ea5fff15-2db2-4317-b16b-d1a277ba6424',  -- 2026-07-21 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: NCERT Class XI, Cha
  'd874fa5c-3e2e-4bd4-b35a-ff12940eec7f',  -- 2026-07-21 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: NCERT Class XII, Ch
  '39c53ef1-9258-483d-9aca-2409382ec0c1',  -- 2026-07-21 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: NCERT Class XI, Cha
  '025ad1a2-e5a9-4b5a-83e2-c95a956cce73',  -- 2026-07-21 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: NCERT Class XII, Ch
  '1442309f-aa8b-4bf1-a818-90a59f38c453',  -- 2026-07-21 | **NEET महत्व: ⭐⭐⭐⭐⭐ (5/5)** /  / 📚 **अध्याय:** NCERT कक्षा XI, अ
  '74591c02-fe31-4264-ad67-95ad08968186',  -- 2026-07-21 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: NCERT Class 11, Cha
  '5acd377d-ab45-4ffe-8b7c-22b011b33dad',  -- 2026-07-21 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: NCERT Class XI, Cha
  '26cc60bb-3874-4e99-a736-4995c7e1a9a2',  -- 2026-07-21 | Hello! 👋  /  / I'm **NEET-AI**, your expert tutor for Indian med
  '49424d06-e6ec-4288-9885-9db3e31f0833',  -- 2026-07-22 | NEET Importance: ⭐⭐⭐⭐ (4/5) /  / 📚 Chapter: NCERT Class XI, Chap
  '673f7656-546a-48c3-8863-91cadc0eedbb',  -- 2026-07-25 | Great! Let me reveal the answer and explanation, Student. 🎯 / 
  '7cce4488-061e-45cd-8397-327af3e9d1cb',  -- 2026-07-25 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: NCERT Class XI, Cha
  '19b6042d-eb03-43d1-b9ad-52fe36557801',  -- 2026-07-26 | **NEET Importance: ⭐⭐⭐⭐⭐ (5/5)** /  / 📚 **Chapter:** NCERT Class
  'e98c0255-1501-4205-ae46-ef84e2b303d4',  -- 2026-07-26 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: NCERT Class XI, Cha
  'bd8a408c-6fc3-4799-b723-06504037d6f1',  -- 2026-07-26 | NEET Importance: ⭐⭐⭐⭐ (4/5) /  / 📚 Chapter: [NCERT Class XI, Cha
  '1206a534-13f5-42ba-b36a-cf0540d7d911',  -- 2026-07-26 | NEET Importance: ⭐⭐⭐⭐ (4/5) /  / 📚 Chapter: [NCERT Class XI, Cha
  'cc1571db-cd22-4150-92db-d7742637a7b3',  -- 2026-07-26 | NEET Importance: ⭐⭐⭐⭐ (4/5) /  / 📚 Chapter: NCERT Class 11, Chap
  '86757444-035d-4b4a-8db4-a91cca4f44e7',  -- 2026-07-26 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: NCERT Class XI, Cha
  '60f1ea09-cdb1-475b-9a13-677897bb25ef',  -- 2026-07-26 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: NCERT Class XI, Cha
  '4d653f53-a2e4-4950-a771-876d43b366be',  -- 2026-07-27 | # Fetal Development Flowchart /  / I'd be happy to help, Student
  '52e19aaf-fa7c-4723-91c5-dfffb1775bf9',  -- 2026-07-27 | # Additional Fetal Development Flowcharts /  / Sure, Student! He
  '2647d35a-8ee1-442c-83fc-ae60dce41e6e',  -- 2026-07-27 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: NCERT Class 11 — Bi
  'bdddfb50-513f-4656-842b-8366a2fddcfc',  -- 2026-07-29 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: NCERT Class 11, Cha
  '9fd6c421-6fe4-425d-9fde-d3ac0a35448c'   -- 2026-07-29 | NEET Importance: ⭐⭐⭐⭐⭐ (5/5) /  / 📚 Chapter: NCERT Class 11, Cha
);

-- STEP 2 (recommended): every remaining pre-versioning row -- unreachable from now on, since the
-- new key includes the prompt version, language and answer style. Expect 57 before deleting.
select count(*) from public.answer_cache where prompt_version is null;
delete from public.answer_cache where prompt_version is null;
